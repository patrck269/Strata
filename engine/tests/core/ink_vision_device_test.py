"""Exercise native CPU image dispatch and forced-GPU rejection with real external weights."""
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.parse


def main():
    if len(sys.argv) != 4:
        print('SKIP: supply native engine, native model pack, and actual vision shard')
        return 77
    exe, model, vision = sys.argv[1:]
    bad = subprocess.run([exe, '--vision-device', 'invalid'], capture_output=True, text=True)
    assert bad.returncode == 2 and '--vision-device' in bad.stderr, bad
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / 'gradient.bmp'
        width = height = 40
        pixels = b''.join(bytes((x*255//39, y*255//39, 200)) for y in range(height) for x in range(width))
        header = b'BM' + struct.pack('<IHHI', 54+len(pixels), 0, 0, 54)
        header += struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, len(pixels), 0, 0, 0, 0)
        image.write_bytes(header + pixels)
        for mode in ('gpu', 'cpu', 'auto'):
            cmd = [exe, '--cpu', '--serve', '--native', model, '--vision', vision, '--vision-device', mode,
                   '--max-context', '512', '--parallel', '1', '--draft', 'off', '--threads', '8', '--cpu-threads', '8']
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            lines, errors = queue.Queue(), []
            def stdout():
                for line in p.stdout:
                    lines.put(line.rstrip())
                lines.put(None)
            def stderr():
                for line in p.stderr:
                    errors.append(line.rstrip())
            t1, t2 = threading.Thread(target=stdout), threading.Thread(target=stderr)
            t1.start(); t2.start()
            observed = []
            def until(prefixes):
                while True:
                    line = lines.get(timeout=600)
                    assert line is not None, (mode, 'engine exited', observed, errors)
                    observed.append(line)
                    if line.startswith(prefixes):
                        return line
            def send(line):
                p.stdin.write(line+'\n'); p.stdin.flush()
            try:
                ready = until(('READY ', 'ERR '))
                assert ready.startswith('READY '), (mode, ready, errors)
                send('GEN 1 image='+urllib.parse.quote(str(image), safe='')+' 200054,121')
                outcome = until(('DONE ', 'ERR '))
                if mode == 'gpu':
                    assert outcome.startswith('ERR ') and 'forced GPU vision failed' in outcome, (outcome, errors)
                    assert not any('VISION_ENCODE device=cpu' in s for s in errors), errors
                    send('GEN 1 121,122')
                    resumed = until(('DONE ', 'ERR '))
                    assert resumed.startswith('DONE ') and int(resumed.split()[1]) == 1, (resumed, errors)
                else:
                    assert outcome.startswith('DONE ') and int(outcome.split()[1]) == 1, (outcome, errors)
                    assert any(s.startswith('IMG 0 1 1 1') for s in observed), observed
                send('QUIT')
                p.wait(timeout=30)
                t1.join(timeout=5); t2.join(timeout=5)
                assert p.returncode == 0, (p.returncode, errors)
                if mode != 'gpu':
                    assert any(s.startswith('VISION_ENCODE device=cpu ') and 'tokens=1 ' in s for s in errors), errors
                    assert not any(s.startswith('VISION_LOAD device=cuda') or s.startswith('VISION_LOAD device=hip') for s in errors), errors
                print('VISION_DEVICE mode='+mode+' forced_cpu_backend=PASS')
            finally:
                if p.poll() is None:
                    p.kill(); p.wait(timeout=30)
                t1.join(timeout=5); t2.join(timeout=5)
                p.stdin.close(); p.stdout.close(); p.stderr.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

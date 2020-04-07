from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import argparse
import numpy as np
import cgi
import cv2
import dlib

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        res = ''
        if ctype == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            pdict['CONTENT-LENGTH'] = self.headers.get('content-length')
            fields = cgi.parse_multipart(self.rfile, pdict)
            print(fields.keys())
            try :
                image = cv2.imdecode(np.frombuffer(fields['photo'][0], dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite("new.jpg",image)
                detections = detector(image, 1)
                for h, b in enumerate(detections):
                    shape = sp(image, b)
                    face_chip = dlib.get_face_chip(image, shape)
                    face_descriptor = facerec.compute_face_descriptor(face_chip)
                    face_vec = np.zeros(shape=128)
                    for i in range(0, len(face_descriptor)):
                        face_vec[i] = face_descriptor[i]
                if fields['method'][0] == 'recognizePerson':
                    #print('Here')
                    res = 'FACE RECOGNIZE'
                elif fields['method'][0] == 'addPerson':
                    res = 'OK'
            except KeyError:
                res = "ERROR: Unknown request"
        self.wfile.write(res.encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def run(server_class=ThreadedHTTPServer, handler_class=Handler, addr="", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9999,
        help="Specify the port on which the server listens",
    )

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
"""Draws squares around detected faces in the given image."""

import argparse

from google.cloud import vision
from PIL import Image, ImageDraw

def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.
    Args:
        face_file: A file-like object containing an image with faces.
    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.Image(content=content)

    return client.face_detection(
        image=image, max_results=max_results).face_annotations

def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.
    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save(output_filename)


def main(input_filename, output_filename, max_results):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(output_filename))

        image.seek(0)
        highlight_faces(image, faces, output_filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detects faces in the given image.')
    parser.add_argument(
        'input_image', help='the image you\'d like to detect faces in.')
    parser.add_argument(
        '--out', dest='output', default='out.jpg',
        help='the name of the output file.')
    parser.add_argument(
        '--max-results', dest='max_results', default=4,
        help='the max results of face detection.')
    args = parser.parse_args()

    main(args.input_image, args.output, args.max_results)

Image.open("out.jpg")

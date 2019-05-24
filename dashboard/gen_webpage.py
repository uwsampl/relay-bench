"""Generates a webpage for visualizing benchmark output data.

The "graph" and "raw_data" folders in the output directory must already exist
and be populated before running this script.
"""
import argparse
import os

from PIL import Image

PAGE_PREFIX = '''
<hmtl>
<head>
<title>TVM Relay Dashboard</title>
</head>
<body bgcolor="ffffff" link="006666" alink="8b4513" vlink="006666" style="background-image: url(%s); background-position: center;">
<div align="center">
<div align="center"><h1 style="background-color: white;">TVM Relay Dashboard</h1></div>
'''

PAGE_SUFFIX = '''
</div>
</body>
</html>
'''

GRAPH_DIR_PATH='graph'
LORD_JERRY_PATH='jerry.jpg'


def get_img_dims(img_path, scale=1.0):
    img = Image.open(img_path)
    w, h = img.size
    return w * scale, h * scale


def create_website(out_dir, img_paths, bg_image):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    page_body = ''
    for img_path in img_paths:
        w, h = get_img_dims(img_path, scale=0.75)
        page_body += f'<img src="{img_path}" style="width:{w}px;height:{h}px;padding:10px;">\n'

    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(PAGE_PREFIX % bg_image)
        f.write(page_body)
        f.write(PAGE_SUFFIX)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)

    args = parser.parse_args()
    out_dir = os.path.join(os.getcwd(), args.out_dir)
    # Switch to the output directory, so we don't need to keep track of
    # separate paths for loading images while the script is running and loading
    # images when viewing the generated webpage.
    os.chdir(out_dir)
    img_paths = []
    for filename in os.listdir(GRAPH_DIR_PATH):
        if filename.endswith('.png'):
            img_paths.append(os.path.join(GRAPH_DIR_PATH, filename))
    bg_image = LORD_JERRY_PATH
    create_website(out_dir, img_paths, bg_image)

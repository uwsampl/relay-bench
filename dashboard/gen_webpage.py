import argparse
import os

from PIL import Image

PAGE_PREFIX = '''
<hmtl>
<head>
<title>TVM Relay Dashboard</title>
</head>
<body bgcolor="ffffff" link="006666" alink="8b4513" vlink="006666">
<div align="center">
<div align="center"><h1>TVM Relay Dashboard</h1></div>
'''

PAGE_SUFFIX = '''
</div>
</body>
</html>
'''

class DualPath:
    '''A representation of paths for both the working directory and the web.'''

    def __init__(self, cwd_prefix, web_prefix, file_name):
        self.cwd_prefix = cwd_prefix
        self.web_prefix = web_prefix
        self.file_name = file_name

    def get_cwd_path(self):
        return os.path.join(self.cwd_prefix, self.file_name)

    def get_web_path(self):
        return os.path.join(self.web_prefix, self.file_name)


def get_img_dims(img_path, scale=1.0):
    img = Image.open(img_path)
    w, h = img.size
    return w * scale, h * scale


def create_website(output_dir, img_paths):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    page_body = ''
    for img_path in img_paths:
        w, h = get_img_dims(img_path.get_cwd_path(), scale=0.5)
        page_body += f'<img src="{img_path.get_web_path()}" style="width:{w}px;height:{h}px;padding:10px;">\n'

    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(PAGE_PREFIX)
        f.write(page_body)
        f.write(PAGE_SUFFIX)


if __name__ == '__main__':
    description = 'Generate a webpage for visualizing benchmark output data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output-dir', default='web')

    args = parser.parse_args()
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    img_paths = []
    for i in range(4):
        img_paths.append(DualPath('graphs', '../graphs', 'graph.png'))
    create_website(output_dir, img_paths)

"""Generates a webpage for visualizing benchmark output data.

The "graph" and "raw_data" folders in the output directory must already exist
and be populated before running this script.
"""
import argparse
import json
import os

from PIL import Image

PAGE_PREFIX_TEMPLATE = '''
<hmtl>
<head>
<title>TVM Relay Dashboard</title>
</head>
<body bgcolor="ffffff" link="006666" alink="8b4513" vlink="006666" style="background-image: url(%s); background-position: center;">
<div align="center">
<div align="center"><h1 style="background-color: white;">TVM Relay Dashboard</h1></div>
%s
'''

PAGE_SUFFIX_TEMPLATE = '''
</div>
<script>
function elem(id) {
  return document.getElementById(id);
}

function install_countdown(nm, tgt) {
  // update every second
  var x = setInterval(function() {
    var diff = tgt - (new Date().getTime());

    var d = Math.floor(diff / (1000 * 60 * 60 * 24));
    var h = Math.floor((diff %% (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    var m = Math.floor((diff %% (1000 * 60 * 60)) / (1000 * 60));
    var s = Math.floor((diff %% (1000 * 60)) / 1000);

    var tm = "";
    if(d > 0) {
      tm = d + " days";
    } else if(h > 0) {
      tm = h + " hours";
    } else if(m > 0) {
      tm = m + " minutes";
    } else if(s > 0) {
      tm = s + " seconds";
    } else {
      tm = tgt + " passed";
    }
    elem(nm).innerHTML = tm;

    if(diff < 0) {
      // how can this work if x not in scope?!
      // https://www.w3schools.com/howto/howto_js_countdown.asp
      clearInterval(x);
    }
  }, 1000);
}

window.onload = function(e) {
  %s
}
</script>
</body>
</html>
'''

LORD_JERRY_PATH='jerry.jpg'

def get_img_dims(img_path, scale=1.0):
    img = Image.open(img_path)
    w, h = img.size
    return w * scale, h * scale


def init_page_prefix_template(config):
    # Create a countdown header for every deadline in the config.
    deadline_html = ''
    if 'deadlines' in config:
        for deadline_name in config['deadlines']:
            deadline_html += '<div align="center"><h2 style="background-color: white; color: red;">%s: <span id="%s"></span></h1></div>\n' % (deadline_name, deadline_name)
    return PAGE_PREFIX_TEMPLATE % (LORD_JERRY_PATH, deadline_html)


def init_page_suffix_template(config):
    # Install a countdown for every deadline in the config.
    deadline_html = ''
    if 'deadlines' in config:
        deadlines = config['deadlines']
        for deadline_name in deadlines:
            deadline_html += '  install_countdown("%s", new Date("%s").getTime());\n' % (
                deadline_name, deadlines[deadline_name]['date'])
    return PAGE_SUFFIX_TEMPLATE % deadline_html


def create_website(out_dir, img_paths, config):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    page_body = ''
    for img_path in img_paths:
        w, h = get_img_dims(img_path, scale=0.75)
        page_body += f'<img src="{img_path}" style="width:{w}px;height:{h}px;padding:10px;">\n'

    page_prefix = init_page_prefix_template(config)
    page_suffix = init_page_suffix_template(config)
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(page_prefix)
        f.write(page_body)
        f.write(page_suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir', type=str, required=True,
                        help='Directory where graphs are found, should be absolute')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Directory where output should be created, should be absolute')
    parser.add_argument('--config-dir', type=str, default='',
                        help='Directory to look for a config.json file')
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, 'config.json')) as json_file:
        config = json.load(json_file)

    out_dir = args.out_dir
    # Switch to the output directory, so we don't need to keep track of
    # separate paths for loading images while the script is running and loading
    # images when viewing the generated webpage.
    os.chdir(out_dir)
    img_paths = []
    for filename in os.listdir(args.graph_dir):
        # graphs should be indexed relatively: (site root)/graph/*.png
        if filename.endswith('.png'):
            img_paths.append(os.path.join('graph', filename))
    create_website(out_dir, img_paths, config)

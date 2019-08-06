"""Generates a webpage for visualizing benchmark output data.

The "graph" and "raw_data" folders in the output directory must already exist
and be populated before running this script.
"""
import argparse
import json
import os
import shutil

from PIL import Image

PAGE_PREFIX_TEMPLATE = '''
<hmtl>
<head>
<title>TVM Relay Dashboard</title>
</head>
%s
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
      tm = "passed";
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

def main(out_dir, graph_dir, dash_home_dir):
    with open(os.path.join(dash_home_dir, 'config.json')) as json_file:
        main_config = json.load(json_file)
    exp_config = get_exp_config(dash_home_dir)

    set_up_out_dir(out_dir, graph_dir, main_config)
    # Switch to the output directory, so we don't need to keep track of
    # separate paths for loading images while the script is running and loading
    # images when viewing the generated webpage.
    os.chdir(out_dir)

    page_prefix = init_page_prefix_template(main_config)
    page_body = gen_page_body(exp_config)
    page_suffix = init_page_suffix_template(main_config)
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(page_prefix)
        f.write(page_body)
        f.write(page_suffix)


def gen_page_body(exp_config):
    page_body = ''
    for (curr_dir, _, files) in os.walk('./graph'):
        # Remove the './graph' prefix from the directory path we actually show.
        shown_dir = os.sep.join(curr_dir.split(os.sep)[2:])
        depth = len(shown_dir.split(os.sep))
        section_heading_size = max(2, min(depth, 6))
        if depth == 1 and len(shown_dir) != 0:
            heading_text = exp_config[shown_dir]['title']
        else:
            heading_text = shown_dir
        page_body += f'<h{section_heading_size} style="background-color: white;">{heading_text}</h{section_heading_size}>\n'
        for filename in files:
            if not filename.endswith('.png'):
                continue
            img_heading_size = min(section_heading_size + 1, 6)
            img_path = os.path.join(curr_dir, filename)
            page_body += f'<img src="{img_path}" style="height:400px;padding:10px;">\n'
    return page_body


def init_page_prefix_template(config):
    # Create a countdown header for every deadline in the config.
    deadline_html = ''
    if 'deadlines' in config:
        for deadline_name in config['deadlines']:
            deadline_html += '<div align="center"><h2 style="background-color: white; color: red;">%s: <span id="%s"></span></h1></div>\n' % (deadline_name, deadline_name)
    if 'jerry_path' in config:
        background_html = f'<body bgcolor="ffffff" link="006666" alink="8b4513" vlink="006666" style="background-image: url({LORD_JERRY_PATH}); background-position: center;">'
    else:
        background_html = ''

    return PAGE_PREFIX_TEMPLATE % (background_html, deadline_html)


def init_page_suffix_template(config):
    # Install a countdown for every deadline in the config.
    deadline_html = ''
    if 'deadlines' in config:
        deadlines = config['deadlines']
        for deadline_name in deadlines:
            deadline_html += '  install_countdown("%s", new Date("%s").getTime());\n' % (
                deadline_name, deadlines[deadline_name]['date'])
    return PAGE_SUFFIX_TEMPLATE % deadline_html


def set_up_out_dir(out_dir, graph_dir, config):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    web_graph_dir = os.path.join(out_dir, 'graph')
    shutil.rmtree(web_graph_dir, ignore_errors=True)
    shutil.copytree(graph_dir, web_graph_dir)
    if 'jerry_path' in config:
        shutil.copy(os.path.expanduser(config['jerry_path']), os.path.join(out_dir, LORD_JERRY_PATH))


def get_exp_config(dash_home_dir):
    exp_config = {}
    exp_config_dir = os.path.join(dash_home_dir, 'config')
    for exp_name in os.listdir(exp_config_dir):
        with open(os.path.join(exp_config_dir, exp_name, 'config.json'), 'r') as f:
            exp_config[exp_name] = json.load(f)
    return exp_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir', type=str, required=True,
                        help='Directory where graphs are found, should be absolute')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Directory where output should be created, should be absolute')
    parser.add_argument('--dash-home-dir', type=str, default='',
                        help='Dashboard home directory')
    args = parser.parse_args()

    main(args.out_dir, args.graph_dir, args.dash_home_dir)

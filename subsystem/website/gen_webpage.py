"""Generates a webpage for visualizing benchmark output data.

The "graph" and "raw_data" folders in the output directory must already exist
and be populated before running this script.
"""
import argparse
import json
import os
import shutil

from common import (read_config, write_status, idemp_mkdir, 
                    render_exception)
from dashboard_info import DashboardInfo

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

def main(config_dir, home_dir, out_dir):
    config = read_config(config_dir)
    info = DashboardInfo(home_dir)
    exp_titles = get_exp_titles(info)

    deadline_config = None
    if info.subsys_config_valid('deadline'):
        deadline_config = info.read_subsys_config('deadline')

    set_up_out_dir(info, out_dir)
    # Switch to the output directory, so we don't need to keep track of
    # separate paths for loading images while the script is running and loading
    # images when viewing the generated webpage.
    os.chdir(out_dir)

    page_prefix = init_page_prefix_template(deadline_config)
    page_body = gen_page_body(info, exp_titles)
    page_suffix = init_page_suffix_template(deadline_config)
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(page_prefix)
        f.write(page_body)
        f.write(page_suffix)
    write_status(out_dir, True, 'success')


def gen_page_body(info, exp_titles):
    page_body = ''
    for (curr_dir, _, files) in os.walk(info.exp_graphs):
        # Remove the './graph' prefix from the directory path we actually show.
        shown_dir = os.sep.join(curr_dir.split(os.sep)[2:])
        depth = len(shown_dir.split(os.sep))
        section_heading_size = max(2, min(depth, 6))
        heading_text = shown_dir
        if depth == 1 and len(shown_dir) != 0 and shown_dir in exp_titles:
            heading_text = exp_titles[shown_dir]
        page_body += f'<h{section_heading_size} style="background-color: white;">{heading_text}</h{section_heading_size}>\n'
        for filename in files:
            if not filename.endswith('.png'):
                continue
            img_heading_size = min(section_heading_size + 1, 6)
            img_path = os.path.join(curr_dir, filename)
            page_body += f'<img src="{img_path}" style="height:400px;padding:10px;">\n'
    return page_body


def init_page_prefix_template(deadline_config):
    # Create a countdown header for every deadline in the config.
    deadline_html = ''
    if deadline_config is not None and 'deadlines' in deadline_config:
        for deadline_name in deadline_config['deadlines']:
            deadline_html += '<div align="center"><h2 style="background-color: white; color: red;">%s: <span id="%s"></span></h1></div>\n' % (deadline_name, deadline_name)

    background_html = f'<body bgcolor="ffffff" link="006666" alink="8b4513" vlink="006666" style="background-image: url({LORD_JERRY_PATH}); background-position: center;">'

    return PAGE_PREFIX_TEMPLATE % (background_html, deadline_html)


def init_page_suffix_template(deadline_config):
    # Install a countdown for every deadline in the config.
    deadline_html = ''
    if deadline_config is not None and 'deadlines' in deadline_config:
        deadlines = deadline_config['deadlines']
        for deadline_name in deadlines:
            deadline_html += '  install_countdown("%s", new Date("%s").getTime());\n' % (
                deadline_name, deadlines[deadline_name]['date'])
    return PAGE_SUFFIX_TEMPLATE % deadline_html


def set_up_out_dir(info, out_dir):
    idemp_mkdir(out_dir)

    web_graph_dir = os.path.join(out_dir, 'graph')
    shutil.rmtree(web_graph_dir, ignore_errors=True)
    shutil.copytree(info.exp_graphs, web_graph_dir)
    shutil.copy(os.path.abspath('jerry.jpg'),
                os.path.join(out_dir, LORD_JERRY_PATH))


def get_exp_titles(info):
    exp_titles = {}
    for exp_name in info.all_present_experiments():
        if info.exp_config_valid(exp_name):
            conf = info.read_exp_config(exp_name)
            if 'title' in conf:
                exp_titles[exp_name] = conf['title']
    return exp_titles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=str, required=True,
                        help='Directory where graphs are found, should be absolute')
    parser.add_argument('--home-dir', type=str, required=True,
                        help='Dashboard home directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where output should be created, should be absolute')
    args = parser.parse_args()

    main(args.config_dir, args.home_dir, args.output_dir)
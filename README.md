# Relay Dashboard Documentation

## Foreword

The Relay Dashboard is intended as a flexible infrastructure for adding Relay experiments and analyses to be run every night and for the results to be visualized. It was developed over the course of preparing research publications for the Relay IR, with the goal of ensuring that our experiments would always be in working order and that at any given moment, we had a graph of our latest experiments to draw upon. This is to say that the dashboard's development was rather ad hoc and under the time pressure of impending publication deadlines. In making this infrastructure public, the original authors of the dashboard hope that the hasty design decisions of the past can be recognized and improved through public development and that the dashboard can be useful to other TVM research and development efforts.

## Core Concepts

The dashboard is, in essence, a cron job that is responsible for ensuring that experiments' dependencies are met, then running experiments and taking measurements, and creating graphs and a website in order to read their results. We divide this functionality in terms of experiments and subsystems, with experiments containing the code responsible for producing and analyzing data and subsystems handling the matter of displaying the results. The dashboard infrastructure is responsible for keeping configuration information for experiments and subsystems and for activating each experiment and subsystem in turn.

### Top-Level Dashboard

The core dashboard infrastructure is implemented in the folder dashboard (other parameters and configurations will be discussed later) and can be activated through the script `run_dashboard.sh`, which takes as a parameter a "home directory." The dashboard's home directory is where all information about experiment and subsystem configurations and results is stored, as well as the configuration for the dashboard itself.

The core infrastructure is responsible for doing the following:
- Building TVM (pulling in a fresh master and checking out branches if experiments request a specific one) and pulling in the Relay AOT compiler
- Checking that the home directory is well-formed
- Detecting experiments and subsystems based on whether configurations for them exist in the home directory
- "Pre-checking" experiment and subsystem configs: Any that are missing mandatory fields or do not parse are rejected and not run
- Running each experiment according to the protocol defined later in this document
- Running each subsystem according to the protocol defined later in this document
- Making an archive of the dashboard home before and after this run

The dashboard home directory organization is as follows:
- `config`
   * `config.json`: Top-level configuration information for the dashboard infrastructure
   * `experiments`: A subdirectory corresponding to each experiment (subdirectory) in the repo's experiments directory, matching the name. Each subdirectory contains a `config.json` with that experiment's configuration.
   * subsystem: A subdirectory corresponding to each subsystem (subdirectory) in the repo's subsystem directory, matching the name. Each subdirectory contains a `config.json`  with that subsystem's configuration.
- `results`
  * `experiments`: Each subdirectory below contains the results of running each experiment, with a subdirectory corresponding to each experiment (subdirectory) in the repo's `experiments` directory, matching the name.
    - `status`: contains the `status.json` files produced by each stage of running the experiment, with the name of the file matching the stages (`precheck.json`, `run.json`, etc.).
    - `data`: contains the `data.json` files produced by the analysis stage of each experiment. This subdirectory will contain all the `data.json` files from all runs of the experiment, so there is a timestamp corresponding to the time the dashboard started running appended to the filenames (as well as in a field within the JSON files).
    - `summary`: Contains the `summary.json` file produced by the summary stage of each experiment. Only keeps the one from the latest run of the dashboard.
    - `graph`: Contains the graph files produced by the visualization stage of each experiment. Only keeps those from the latest run of the dashboard.
  * `subsystem`: Analogous to that for experiments
    - `status`: contains the `status.json` files produced by each stage of running the subsystem, with the name of the file matching the stages (only `precheck.json` and `run.json`).
    - `output`: Each subsystem can manage its own output subdirectory as it wishes and the contents are not deleted between runs of the dashboard.

The top-level dashboard config.json may contain the following fields:
- `tmp_data_dir` (str, mandatory): Directory for storing experiment raw data (we hope to move this to cloud storage eventually), which are zipped CSV files
- `backup_dir` (str, mandatory): Directory for storing compressed copies of dashboard backups AKA dumping zip files (we hope to move this to cloud storage too)
- `setup_dir` (str, mandatory): Directory for storing persistent setup files for experiments (this probably should stay local)
- `run_cpu_telemetry` (boolean, optional): Top-level switch for CPU logging for all experiments (can be overwritten by configurations of experiments, default false)
- `run_gpu_telemetry` (boolean, optional): Top-level switch for GPU logging for all experiments (can be overwritten by configurations of experiments, default false)
- `telemetry_rate` (integer, optional): The rate (in seconds) that the telemetry process collect data from `sensors` and `nvidia-smi` (e.g. setting to 30 will make the telemetry process collect data once 30 seconds). The default value is 15. To disable the telemetry process, set this field to a negative integer.
- `randomize` (boolean, optional): Whether to randomize the experiment order. Defaults to true. If false, experiments will be run based on their specified priority (ties broken by lexicographic order by name).

Example configurations for the dashboard and every experiment and subsystem are given in `sample-dashboard-home/`.

### Experiments

Experiments are primarily responsible for collecting and analyzing data. Their implementations live in the experiments subdirectory of the repo.

*Note: Our design also asks that experiments produce a human-readable text summary and their own graphs, which makes for a less clean abstraction. The original reason for this is that subsystems were introduced late in our design and the Slack integration and website were previously hardcoded into the infrastructure; there probably should be subsystems for handling summaries and graphs and some way to specify via configuration how to turn analyzed data into summaries and graphs, but that would take a lot of redesigning.*

Each experiment subdirectory in the repo should contain the following bash scripts, corresponding to a "stage" of the experiment (run in this order):
- `run.sh`: This script takes in a config directory (directory containing the experiment's  `config.json`) and a destination directory to dump raw data. It uses the information in the `config.json` file to configure and run the experiment itself, generating raw data.
- `analyze.sh`: This script takes in the experiment's config directory, a source directory containing raw data, and a destination directory and produces a `data.json` file in the destination directory containing summary statistics for the raw data. Note: the dashboard infrastructure appends a "`timestamp`" field (containing a string expressing the time at which analyze.sh was run to produce that file); a "`tvm_hash`" field indicating the TVM commit used to run that experiment to the `data.json` file after this script runs; and "`start_time`", "`end_time`", and "`time_delta`" fields that record the time the experiment's start time, end time, and duration.
- `visualize.sh`: This script takes in the experiment's config directory and a source directory containing one or more JSON files of the form produced by `analyze.sh`. The script outputs one or more graphs using the files in the source directory to the destination directory; it is up to the script whether or not to produce longitudinal graphs. Note that `visualize.sh` can organize its destination directory however it likes, e.g., creating subdirectories corresponding to different categories of graphs.
- `summarize.sh`: Analogous to `visualize.sh`, but produces a text summary as a `summary.json` in its destination directory. The JSON file takes the form of a Slack message "item" with only two fields: "`title`" and "`value`" (the former is self-explanatory; the latter consists of the text summary)

Optional script (runs first if present):
- `setup.sh`: This script should perform any first-time setup for an experiment. It takes in a source directory containing a `config.json` and a destination directory where any files that need to be produced should be deposited. The dashboard infrastructure will normally only rerun `setup.sh` if the experiment's directory has changed since the last run **(possibly dangerous hack: the dashboard determines this by using git to find the date of last edit, so this implicitly means that experiment source code has to live in a git repository)**. After `setup.sh` has run (or if it has run before and does not need to be run again), the files it produced in its `setup` directory will be copied into the experiment directory in a directory called "`setup`" before any of the experiment's other functions have run. *(Note: This was implemented very hastily so certain experiments did not have to download multiple-GB data files each run and incur possible network failures. There is probably a cleaner possible design.)*

Each of these scripts should emit a `status.json` file in its destination directory as well. This file should contain a boolean "`success`" field to indicate whether the script succeeded and a "`message`" field detailing any errors that occurred in that stage. The dashboard infrastructure relies on these to determine whether a stage succeeded (if any of the scripts terminates without leaving a `status.json`, the infrastructure assumes that stage failed).

Experiment `config.json` files may contain, in addition to any fields specific to that experiment, the following special fields used by the dashboard framework itself:
- `active` (mandatory, boolean): whether to run the benchmark. 
- `title` (optional, string): a name for the benchmark
- `description` (optional, string):  describes the experiment
- `notify` (optional, array of strings): slack IDs of anyone who should be pinged if the experiment fails
- `tvm_remote` (optional, string): TVM fork to use for tvm_branch's functionality
- `tvm_branch` (optional, string): If indicated, the experiment will check out the specified branch from the `tvm_remote` repo and build that variant of TVM for the experiment
- `rerun_setup` (optional, boolean): If indicated and the experiment has a `setup.sh`, this will force the setup to be rerun regardless of whether the experiment has changed. Defaults to false.
- `process_pinning` (optional, dict): configuration of process pinning for experiments (using `taskset`)
  * `enable` (mandatory, boolean): Switch for the process pinning
  * `cores`: (mandatory, parameter passed to `taskset`): Bitmask / cpu list, etc. See `man taskset` for more information.
  * Example `process_pinning` dictionary: `"process_pinning": {"enable": true, "cores": "0-7"}`
- `run_cpu_telemetry` (optional, boolean): Switch of CPU logging for current experiment. If indicated, the configuration will overwrite the top-level configuration for current experiment. (default: same as the value in top-level configuration).
- `run_gpu_telemetry` (optional, boolean): Switch of GPU logging for current experiment. If indicated, the configuration will overwrite the top-level configuration for current experiment. (default: same as the value in top-level configuration).
- `telemetry_rate` (optional, integer): If indicated, the number in this field will overwrite the timespan between two data collections of the telemetry process, else, the value will be that in the top-level dashboard configuration. 
- `priority` (optional, int): If the dashboard is not set to run experiments in random order, the priority will be used to decide the experiment ordering. If unspecified, the priority will default to 0. The highest-priority experiments will run first. Ties will be broken by lexicographic order by experiment directory name. (This mechanism is included primarily for debugging purposes, like determining if the experiment ordering affects the results. Experiments should not rely on running in any particular order, however.)

Each script will be executed from its own directory so they don't have to use absolute addresses everywhere.

### Subsystems

Subsystems are nightly tasks that are intended to be executed after all experiments have run. Subsystems have access to all results from all experiments and other subsystems and are meant to be places to have meta-analyses or reporting that is of a scope greater than individual experiments. Subsystems live in the subsystem directory in the benchmark repo.

Each subsystem subdirectory is only required to contain a `run.sh` script that is responsible for running the subsystem's functionality. The script is given the following arguments: a source directory containing a `config.json` file, a target directory for depositing any output files and reporting a status.json (analogous to the experiments' statuses, with a success flag and a message field), and the dashboard home directory. Subsystems are not as structured as experiments and hence are free to read anything in the home directory as they need, including using the results of other subsystems (potentially dangerous and should be used sparingly).

Note that the subsystem output directories, unlike those for experiments, are not cleaned up between dashboard runs -- subsystems are free to store any persistent data they need there, but also must manage them.

Subsystems will have config options as follows:
- `active` (mandatory, boolean), whether to run the subsystem
- `priority` (optional, int), higher priority means the subsystem runs earlier. Default priority is 0. This is included because some subsystems may require the results of other subsystems so they should not be executed until those have run. (This should more properly be an actual dependency system, but this functionality is intended more for "reporting"-type subsystems like the Slack integration, which do not have explicit dependencies but just want to run after everything else has gone.) Ties by priority will be broken by lexicographic ordering by name.
- `title` (optional, string): a name for the subsystem
- `description` (optional, string):  describes the subsystem
- `notify` (optional, array of strings): Slack IDs of anyone who should be pinged if the subsystem fails

#### Important Subsystems
- `deadline`: Posts deadline countdowns to Slack based on its configuration.
- `score`: computes overall scores based on experiment data, produces graphs too
- `website`: Puts experiment graphs and score graphs on a webpage. This should run after the deadline and score systems because it includes their results on the website.
- `exp_reports`: Puts experiment summaries into a Slack message
- `stat_alerts`: Pings users on Slack if experiment results are more than a standard deviation outside their historic mean
- `subsys_reporter`: Reports any failed subsystems; also if a subsystem produces a report.json in its output directory, the reporter will put it into a Slack message. This should be configured to run after all other subsystems, since it requires their results.
- `vis_telemetry`: Produces longitudinal graphs for dashboard telemetry data (see below)

*(Meta-note: Something that became clear in the process of developing the subsystems is that the experiments themselves can be handled as a single subsystem that is configured to run first. This might reduce some duplicated logic in the core infrastructure but would take a lot of engineering effort to properly implement and may not be worthwhile.)*

### Telemetry Record

If the top-level dashboard config has telemetry enabled, a telemetry process runs in the background alongside experiments, periodically (configurable) querying the system about CPU and GPU performance data. The telemetry data is later parsed into JSON files that are stored in `DASHBOARD_HOME/results/subsystem/telemetry/EXP_NAME`, where `DASHBOARD_HOME` and `EXP_NAME` are the dashboard home directory and experiment names, respectively. The `vis_telemetry` subsystem produces readable graphs from the data.

The JSON files for GPU telemetry must contain:
1. A timestamp
2. Topic names mapped to an object that has a `data` field and a `unit` field. The `data` field is a list of pairs where the first element is time elapsed from the beginning of the experiment and the second element is the data collected by the telemetry process. The `unit` field is the unit of the data (a string) if one is provided and `null` otherwise.

The JSON files for CPU telemetry must contain:
1. A timestamp
2. Adapter names mapped to an object whose keys are names of sensors and values to the keys are list of pairs, where the first element is time elapsed from the beginning of the experiment and the second element is the data collected by the telemetry process (numeric).

## Implementation Details

### Dependencies

Python 3.4+, with Python dependencies given in requirements.txt. Pip should be used to install these in whatever environment will invoke the dashboard.

Non-Python dependencies:
- CUDA 10.1 or higher and CuDNN 7.5.0 or higher, as the GPU versions of TVM and other frameworks depend on it (see the `run_dashboard.sh` script)
- Machines for running VTA for the `relay_to_vta` experiment if you plan to run it; see its example config
- To actually view the webpage, a web server like nginx should be configured to treat (dashboard home directory)/results/subsystem/output/website/ as the website root, as the index.html file will be produced there
- Subsystems that post to Slack require a webhook

### Scripts

The script `run_dashboard.sh` is the one responsible for actually invoking the core dashboard infrastructure and setting up the environment for its run. Much of the environment setup is meant for running in the reduced environment of cron, since the cron user does not use the bashrc or other setup information of the user it runs under. A lot of it, like the locations for the dashboard's TVM and Relay AOT compiler installs, is also hardcoded, which is unfortunate and should probably be made more configurable. The script includes options for configuring whether to reinstall TVM (it is useful to turn this off for development, since the build takes time) and for where to look for experiment and subsystem implementations (by default, it will assume it is in the same repo as the dashboard implementation, but users may want to have private or custom experiments).

The script `dashboard_job.sh` provides an example of a script that pulls in the latest `relay-bench` repo and runs it with a specific dashboard home. (This is why the configurations and implementations of experiments and subsystems are meant to be kept in separate locations.) This script could be easily modified to suit the contours of a specific user environment, at least with respect to `run_dashboard`'s command-line options.

### Shared Libraries

The dashboard includes some libraries meant for code reuse under the folder `shared`, meant for code reuse between experiments, etc. The location of the `shared` folder is written by `run_dashboard.sh` into the environment variable `BENCHMARK_DEPS` so experiments can put it in their Python path or reference it.

It is far beyond the scope of this document to go into detail on all the files (mostly utility functions) included in it, so this will instead be a high-level roadmap that notes hacks and areas for further design consideration and development:

- `bash/common.sh` contains some utilities for adding files to the PYTHONPATH and logging stderr
- `python`
    * `common.py`: Intended to contain functions used by nearly everything in the dashboard, mostly wrappers over Python standard library functions. The only complicated functions in here are those for dealing with data.json timestamps and querying for data fields; that may need to be reorganized or redesigned.
    * `dashboard_info.py`: Primarily intended for subsystems to use, implements a data structure that is responsible for keeping track of directories and files inside the dashboard home directory. Useful for querying as to experiment and subsystem statuses. Probably a lot of room for reconsidering its design.
    * `plot_util.py`: Provides a library for building up graphs using MatPlotLib and Seaborn. In the future, it may be desirable to replace the visualization library, hence we are keeping around this wrapper to make that easier to do later.
    * `config_util.py`: Contains a basic functions for determining that certain fields are present in config json files and that config fields fulfill certain prerequisites. This could probably be better designed and made into a DSL akin to `plot_util.py`.
    * `check_prerequisites.py`: Mostly intended for checking subsystem prerequisites. Provides a function that checks that certain experiments have run and have desirable settings in their configs. This could probably also be made a DSL akin to `plot_util.py`, depending on what needs emerge.
    * `trial_util.py`: This is where a lot of very confusing code for timing experiments and recording data in CSV files lives. An upside is that this code is used very widely so further profiling that is added here could be used by many experiments.
    * `analysis_util.py`: Also very confusing code responsible for reading raw data CSV files and producing summary statistics, as it has to follow the format set in `trial_util.py`. The main reason this code is so tangled is that the original dashboard has a lot of old data files hanging around and we have not written code for "migrating" those to any new data format; once this can be done relatively easily, it should be possible to simplify the data representation and also the code in this file.
    * `exp_templates.py`: The least principled part of the shared Python files. This file contains "templates" that implement the basic logic for each stage of an experiment, based on what code tended to be repeated most in practice. Some experiments do not follow these templates because they need extra steps for technical reasons and so have all the "dashboard boilerplate" in full. It may be possible to make these a little bit more general so as to handle those. As messy as this is, this should make it easier to change the dashboard's organization, since there is less code duplication in this manner.
    * `slack_util`: Helper functions for constructing Slack messages and invoking the web API
    * `relay_util`: Functions for invoking TVM's compiler and certain common operations
    * `summary_util`: Basic functions for making generic text displays of common `data.json` configurations

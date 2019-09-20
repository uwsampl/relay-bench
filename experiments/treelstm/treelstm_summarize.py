from validate_config import validate
from exp_templates import summarize_template

if __name__ == '__main__':
    summarize_template(validate, use_networks=False)

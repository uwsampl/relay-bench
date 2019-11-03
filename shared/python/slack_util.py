"""
Python utilities for posting to Slack
"""
import datetime
import json
import logging
import os
import requests
import sys

from common import render_exception

def generate_ping_list(user_ids):
    return ', '.join(['<@{}>'.format(user_id) for user_id in user_ids])


def build_field(title='', value='', tm_start=None, tm_end=None, duration=None, short=False):
    ret = {'value': value, 'short': short}
    if title != '':
        ret['title'] = title
    if tm_start and tm_end and duration:
        ret['value'] = f'_Starts at: {tm_start}_ \n _Ends at: {tm_end}_ \n _Duration: {duration}_\n\n' + ret['value']
    return ret


def build_attachment(title='', fields=None, pretext='', text='', color='#000000'):
    ret = {'color': color}

    if title != '':
        ret['title'] = title
    if fields is not None:
        ret['fields'] = fields
    if pretext != '':
        ret['pretext'] = pretext
    if text != '':
        ret['text'] = text

    return ret


def build_message(text='', pretext='', attachments=None):
    ret = {'text': text}
    if 'pretext' != '':
        ret['pretext'] = pretext
    if attachments is not None:
        ret['attachments'] = attachments

    return ret


def post_message(webhook_url, message):
    """
    Attempts posting the given message object to the
    Slack webhook URL.
    Returns whether it was successful and a message.
    """
    try:
        r = requests.post(webhook_url, json=message)
        return (True, 'success')
    except Exception as e:
        return (False, 'Encountered exception:\n' + render_exception(e))

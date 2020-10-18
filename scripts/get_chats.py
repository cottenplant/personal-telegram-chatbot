import sys
import json
import argparse

from pathlib import Path


def parse_arguments():
    # build parser
    parser = argparse.ArgumentParser(description='Process Telegram chat data.')
    
    parser.add_argument('-i', '--infile', nargs='?', type=argparse.FileType('r'), help='input file path formatted as JSON')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), help='output file path for formatted JSON')
    parser.add_argument('-u', '--user', nargs='?', type=str, default=sys.stdin)

    
    parser.parse_args([user, chats_file_infile, chats_file_output])
    parser.parse_args([])
    
    # verbose = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
    # logging.basicConfig(format='%(message)s', level=verbose[args.v], stream=sys.stdout)
    
    return args


def get_personal_chats(chats_json: dict, user_from='samco'):
    """
    chats_json: JSON-formatted Telegram export with chats parsed into python dict
    user_from: returns messages from given user in chat (default arg='samco')
    returns: dict('id': ['messages']) where 'id' is person with whom chat took place
    """
    chats = chats_json['chats']['list']
    results = {}
    for chat in chats[1:]:
        if chat['type'] == 'personal_chat':
            for message in chat['messages']:
                if message['type'] == 'message' and message['from'] == user_from:
                    results[chat['id']] = [message['text'] for message in chat['messages']]
    return results


def main(user, infile, outfile):
    # open and parse input file
    with open(infile, 'r') as file:
        chats_json = json.load(file)
    
    chats = get_personal_chats(chats_json, user)
    
    with open(outfile, 'w') as output_file_path:
        json.dump(chats, output_file_path)


if __name__ == '__main__':
    # default vars
    user = 'samco'  # argparse
    chats_file_infile = Path('data/input/result.json')
    chats_file_output = Path(f'data/output/{user}.json')
    
    main(user, chats_file_infile, chats_file_output)

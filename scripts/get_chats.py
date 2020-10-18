import sys
import json
import argparse

from pathlib import Path


def parse_arguments():
    # build parser
    parser = argparse.ArgumentParser(description='Process Telegram chat data.')
    parser.add_argument('-i', '--infile', nargs='?', type=argparse.FileType('r'), help='input file path formatted as JSON')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), help='output file path for formatted JSON')
    parser.add_argument('-u', '--user', nargs='?', type=str, help='insert user')
    args = parser.parse_args()

    return args


def get_personal_chats(chats_json, user_from):
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


def main():
    # open and parse input file, output result
    with open(chats_file_infile, 'r') as file:
        chats_json = json.load(file)
    chats = get_personal_chats(chats_json, user)
    with open(chats_file_output, 'w') as output_file_path:
        json.dump(chats, output_file_path)


if __name__ == '__main__':
    # default vars
    user = 'samco'
    chats_file_infile = Path('./data/input/result.json')
    chats_file_output = Path(f'./data/output/{user}.json')

    # parse args
    args = parse_arguments()
    if not args.infile:
        args.infile = chats_file_infile
    if not args.outfile:
        args.outfile = chats_file_output
    if not args.user:
        args.user = user
    
    main()

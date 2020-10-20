import json
import argparse

from chatbot.config.settings import paths_local


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Telegram chat data.')
    parser.add_argument('-i', '--infile', nargs='?', type=argparse.FileType('r'), 
                        help='input file path formatted as JSON')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), 
                        help='output file path for formatted JSON')
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
                    results[chat['id']] = [message['text'] for message in chat['messages'] if type(message['text']) == str]
    
    return results


def main():
    # open and parse input file, wrangle format, and dump JSON result
    with open(paths_local['chats_raw_json'], 'r') as infile:
        chats_json = json.load(infile)
    chats = get_personal_chats(chats_json, paths_local['user'])
    with open(paths_local['chats_processed_json'], 'w') as outfile:
        json.dump(chats, outfile)
    
    # also dump full text version for nlp model
    with open(paths_local['chats_processed_txt'], 'w') as outfile:
        outfile.write('. '.join([message for message in chats.values()][0]))
    

if __name__ == '__main__':
    # parse args - later can use for other users
    args = parse_arguments()
    if not args.infile:
        args.infile = paths_local['chats_raw_json']
    if not args.outfile:
        args.outfile = paths_local['chats_processed_json']
    if not args.user:
        args.user = paths_local['user']
    
    main()

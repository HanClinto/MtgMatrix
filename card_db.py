import faiss
import json
import pickle

# Load FAISS indices
data_file = 'oracle-cards-20231113220154'

db_dir = './data'
card_data = []
cards_by_sfid = {}
faiss_indices = {}
faiss_keys_by_index = {}
faiss_indices_by_key = {}
embeddings_by_sfid = {}

def init_db():
    global card_data
    global cards_by_sfid
    global faiss_indices
    global faiss_keys_by_index
    global faiss_indices_by_key
    global embeddings_by_sfid

    # Load card data
    with open(f'./data/{data_file}.json') as f:
        card_data = json.load(f)
        print(f'Loaded {len(card_data)} cards')
        # Remove every card that has layout 'art_series' or 'token' or 'double_faced_token'
        # Remove every card that has a set_type of 'memorabilia' or 'token' or 'minigame'
        for card in card_data[::-1]:
            if card['layout'] in ['art_series', 'token', 'double_faced_token']:
                card_data.remove(card)
            elif card['set_type'] in ['memorabilia', 'token', 'minigame']:
                card_data.remove(card)

            # For every card, if it doesn't have an image, check to see if a card_face has it. If so, move that up to the parent. Otherwise, remove the card entirely.
            if not 'image_uris' in card:
                if 'card_faces' in card:
                    for card_face in card['card_faces']:
                        if 'image_uris' in card_face:
                            card['image_uris'] = card_face['image_uris']
                            break
                if not 'image_uris' in card:
                    print(f'  {card["name"]} ({card["id"]}) has no image_uris')
                    card_data.remove(card)

        print(f' Culled to {len(card_data)} cards')

        # Create a dictionary of card data by SFID
        cards_by_sfid = {card['id']: card for card in card_data}

    # Load FAISS indices
    tags = ['similar',
            #'duplicate',
            'dupe2',
            'spike',
            'melvin',
            #'vorthos',
            'timmy',
            #'johnny',
            #'flavor'
            ]

    for tag in tags:
        filename = f'{db_dir}/faiss_{data_file}_db_{tag}.index'
        faiss_indices[tag] = faiss.read_index(filename)
        print(f'Loaded FAISS index for {tag}')
        with open(f'{db_dir}/faiss_{data_file}_db_{tag}.keys', 'rb') as f:
            faiss_indices_by_key[tag] = json.load(f)
            faiss_keys_by_index[tag] = {v: k for k, v in faiss_indices_by_key[tag].items()}
            print(f'Loaded FAISS keys for {tag}')
        # ex: embeddings_oracle-cards-20231113220154_db_dupe2.pickle
        with open(f'{db_dir}/embeddings_{data_file}_db_{tag}.pickle', 'rb') as f:
            embeddings_by_sfid[tag] = pickle.load(f)
            print(f'Loaded embeddings for {tag}')

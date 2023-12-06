from flask import Flask, render_template, request, jsonify
import faiss
import json
import pickle
# Import spatial
from scipy import spatial
from datetime import datetime

app = Flask(__name__)

# Load FAISS indices
data_file = 'oracle-cards-20231113220154'

db_dir = './data'

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

faiss_indices = {}
faiss_keys_by_index = {}
faiss_indices_by_key = {}
embeddings_by_sfid = {}

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

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    print(f'Fetching home page of most popular cards in Magic, page #{page}...')
    per_page = 20
    num_results = 100
    start = (page - 1) * per_page
    end = start + per_page

    popular_cards = sorted(card_data, key=lambda x: x.get('edhrec_rank', float('inf')))

    total_pages = num_results // per_page + (1 if num_results % per_page else 0)

    return render_template('index.html', cards=popular_cards[start:end], page=page, total_pages=total_pages)

@app.route('/card/<sfid>')
@app.route('/card/<sfid>/page/<int:page>')
def card(sfid, page=1):
    per_page = 20
    num_results = 100
    if not sfid in cards_by_sfid:
        return 'Card not found', 404
    card = cards_by_sfid[sfid]

    start = (page - 1) * per_page
    end = start + per_page


    related_cards = fetch_related_cards(sfid, start_index = start, end_index=end)  # Adjust this function to return paginated results

    total_pages = num_results // per_page + (1 if num_results % per_page else 0)

    #print(f'Related cards: {related_cards[start:end]}')
    for related_card in related_cards:
        if not 'image_uris' in related_card:
            print(f'  {related_card["name"]} ({related_card["id"]}) has no image_uris')


    return render_template('card.html', card=card, related_cards=related_cards, page=page, total_pages=total_pages)

def cosine_similarity(embedding1, embedding2):
    return spatial.distance.cosine(embedding1, embedding2)

def fetch_related_cards(sfid, start_index = 0, end_index = 100):
    print(f' Fetching related cards for {sfid} [{start_index}:{end_index}]...')

    then = datetime.now()

    result_sfids = {}

    for collection_name in faiss_indices.keys():
        sfids, dists = fetch_related_sfids(sfid, index_key=collection_name, num_results=end_index)
        results = zip(sfids, dists, [collection_name] * len(sfids))
        for sfid, dist, axis in results:
            if sfid in result_sfids:
                if dist > result_sfids[sfid][1]:
                    result_sfids[sfid] = [sfid, dist, axis]
            else:
                result_sfids[sfid] = [sfid, dist, axis]

    # Get a sorted list of results, ordered by distance
    results = sorted(result_sfids.values(), key=lambda x: x[1], reverse=True)

    # Trim the results to the start and end indices
    results = results[start_index:end_index]

    related_cards = []

    for sfid, dist, axis in results:
        if sfid in cards_by_sfid:
            card_copy = json.loads(json.dumps(cards_by_sfid[sfid]))
            card_copy['distance'] = dist
            card_copy['axis'] = axis
            
            related_cards.append(card_copy)
        else:
            print(f'  {sfid} not found in cards_by_sfid')

    print(f'  Populated {len(related_cards)} related cards in {datetime.now() - then}')

    return related_cards


def fetch_related_sfids(sfid, index_key='similar', num_results=100):
    print(f' Fetching related sfids for {sfid} in index "{index_key}"...')

    then = datetime.now()

    # Find the index from the SFID
    if not sfid in faiss_indices_by_key[index_key]:
        print(f'  {sfid} not found in faiss_keys[{index_key}]')
        return [], []
    
    #sfid_embeddings = embeddings_by_sfid[index_key][sfid]

    sfid_index = faiss_indices_by_key[index_key][sfid]
    # Retrieve the embeddings for the SFID
    sfid_embeddings = faiss_indices[index_key].reconstruct(sfid_index)

    # Reshape the embeddings to be a 1xN array
    sfid_embeddings = sfid_embeddings.reshape(1, -1)

    # Find the card with the highest cosine similarity to the embedding_ref
    dists, sfid_indices = faiss_indices[index_key].search(sfid_embeddings, num_results)

    related_dists = dists[0]
    related_sfids = [faiss_keys_by_index[index_key][index] for index in sfid_indices[0]]

    now = datetime.now()
    print(f'  Found {len(related_sfids)} related sfids in {now - then}')

    # Find the card with the highest cosine similarity to the embedding_ref
    return related_sfids, related_dists


if __name__ == '__main__':
    app.run(debug=True)

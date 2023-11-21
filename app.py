from flask import Flask, render_template, request, jsonify
import chromadb
import json
import pickle
# Import spatial
from scipy import spatial
from datetime import datetime

app = Flask(__name__)

# Load Chroma database
db_dir = 'data/oracle-cards-20231113220154_db'

client = chromadb.PersistentClient(path=db_dir)
collections_info = client.list_collections()
collections = {}
for collection_info in collections_info:
    collection_name = collection_info.name
    print(f'Loading collection name: "{collection_name}"')
    if collection_name == "flavor" or collection_name == "timmy" or collection_name == "duplicate":
        continue
    collection = client.get_collection(collection_name)
    collections[collection_name] = collection



#embeddings_similar_file = 'data/embeddings_oracle-cards-20231113220154_db_vorthos.pickle'
embeddings_similar_file = 'data/embeddings_oracle-cards-20231113220154_db_similar.pickle'

# Load collections
print(f'Loading collections from {embeddings_similar_file}...')
with open(embeddings_similar_file, 'rb') as f:
    embeddings_similar = pickle.load(f)
    print(f'Loaded {len(embeddings_similar)} embeddings in collection "similar"')

cached_ids = collections['similar'].get(include=[])['ids']
#cached_ids = list(embeddings_similar.keys())

print(f'Precached {len(cached_ids)} card embeddings')
print(f' First 20: {cached_ids[:20]}')

# Load card data
with open('./data/oracle-cards.json') as f:
    card_data = json.load(f)
    print(f'Loaded {len(card_data)} cards')
    # Remove every card that has layout 'art_series' or 'token' or 'double_faced_token'
    # Remove every card that has a set_type of 'memorabilia' or 'token' or 'minigame'
    for card in card_data[::-1]:
        if card['layout'] in ['art_series', 'token', 'double_faced_token']:
            card_data.remove(card)
            if card['id'] in embeddings_similar:
                del embeddings_similar[card['id']]
        elif card['set_type'] in ['memorabilia', 'token', 'minigame']:
            card_data.remove(card)
            if card['id'] in embeddings_similar:
                del embeddings_similar[card['id']]
#        elif not 'paper' in card['games']:
#            card_data.remove(card)

    # For every card, if it doesn't have an image, check to see if a card_face has it. If so, move that up to the parent. Otherwise, remove the card entirely.
    for card in card_data[::-1]:
        if not 'image_uris' in card:
            if 'card_faces' in card:
                for card_face in card['card_faces']:
                    if 'image_uris' in card_face:
                        card['image_uris'] = card_face['image_uris']
                        break
            
            if not 'image_uris' in card:
                print(f'  {card["name"]} ({card["id"]}) has no image_uris')
                card_data.remove(card)
                if card['id'] in embeddings_similar:
                    del embeddings_similar[card['id']]

    # Remove every card that is not in cached_ids
    card_data = [card for card in card_data if card['id'] in cached_ids]
    print(f' Culled to {len(card_data)} cards')
    # Create a dictionary of card data by SFID
    cards_by_sfid = {card['id']: card for card in card_data}


@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
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

def fetch_related_cards(sfid, index_key='similar', start_index = 0, end_index = 100):
    print(f' Fetching related cards for {sfid} [{start_index}:{end_index}]...')

    then = datetime.now()

    result_sfids = {}

    for collection_name in collections.keys():
        sfids, dists = fetch_related_sfids(sfid, index_key=collection_name, num_results=end_index)
        results = zip(sfids, dists, [collection_name] * len(sfids))
        for sfid, dist, axis in results:
            if sfid in result_sfids:
                if dist < result_sfids[sfid][1]:
                    result_sfids[sfid] = [sfid, dist, axis]
            else:
                result_sfids[sfid] = [sfid, dist, axis]

    # Get a sorted list of results, ordered by distance
    results = sorted(result_sfids.values(), key=lambda x: x[1])

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

    result = collections[index_key].get(ids=[sfid], include=["embeddings"])
    embeddings = result['embeddings'][0]

    # Find the card with the highest cosine similarity to the embedding_ref
    result = collections[index_key].query(
        query_embeddings=embeddings,
        n_results=num_results)

    related_sfids = result['ids'][0]
    related_dists = result['distances'][0]

    return related_sfids, related_dists


def fetch_related_cards2(sfid, index_key='similar'):
    print(f' Fetching related cards for {sfid}...')

    then = datetime.now()

    result = collections[index_key].get(ids=[sfid], include=["embeddings", "metadatas"])
    #print(f' Retrieved embeddings for {sfid} in {datetime.now() - then}: {result}')
    embedding_ref = result['embeddings'][0]

    embedding_ref2 = embeddings_similar[sfid]

    dist_ref = cosine_similarity(embedding_ref, embedding_ref2)
    print(f'  Distance between embeddings: {dist_ref}')

    # Find the card with the highest cosine similarity to the embedding_ref
    embeddings = embeddings_similar.values()
    dists = [cosine_similarity(embedding_ref, embedding) for embedding in embeddings]

    # Zip the SFIDs and distances together
    sfids_and_dists = zip(embeddings_similar.keys(), dists)

    # Sort the SFIDs and distances by distance
    sfids_and_dists = sorted(sfids_and_dists, key=lambda x: x[1])
    
    top_k = 100
    related_sfids_and_dists = sfids_and_dists[:top_k]

    print(f'  Found {len(related_sfids_and_dists)} related cards in {datetime.now() - then}')

    #print(f' Related SFIDs: {related_sfids}')
    related_cards = []
    for sfid, dist in related_sfids_and_dists:
        if sfid in cards_by_sfid:
            card_copy = json.loads(json.dumps(cards_by_sfid[sfid]))
            card_copy['distance'] = dist
            
            related_cards.append(card_copy)
        else:
            print(f'  {sfid} not found in cards_by_sfid')

    print(f'  Populated {len(related_cards)} related cards in {datetime.now() - then}')
    

    return related_cards

if __name__ == '__main__':
    app.run(debug=True)

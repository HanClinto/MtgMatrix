from flask import Flask, render_template, request, jsonify
import chromadb
import json
import pickle
# Import spatial
from scipy import spatial

app = Flask(__name__)

#embeddings_similar_file = 'data/embeddings_oracle_db_similar.pickle'
embeddings_similar_file = 'data/embeddings_oracle_db_duplicate.pickle'

# Load collections
print(f'Loading collections from {embeddings_similar_file}...')
with open(embeddings_similar_file, 'rb') as f:
    embeddings_similar = pickle.load(f)
    print(f'Loaded {len(embeddings_similar)} embeddings in collection "similar"')

cached_ids = list(embeddings_similar.keys())

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
    popular_cards = sorted(card_data, key=lambda x: x.get('edhrec_rank', float('inf')))
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = len(popular_cards) // per_page + (1 if len(popular_cards) % per_page else 0)

    return render_template('index.html', cards=popular_cards[start:end], page=page, total_pages=total_pages)

@app.route('/card/<sfid>')
@app.route('/card/<sfid>/page/<int:page>')
def card(sfid, page=1):
    per_page = 20
    if not sfid in cards_by_sfid:
        return 'Card not found', 404
    card = cards_by_sfid[sfid]

    related_cards = fetch_related_cards(sfid)  # Adjust this function to return paginated results
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = len(related_cards) // per_page + (1 if len(related_cards) % per_page else 0)

    #print(f'Related cards: {related_cards[start:end]}')
    for related_card in related_cards[start:end]:
        if not 'image_uris' in related_card:
            print(f'  {related_card["name"]} ({related_card["id"]}) has no image_uris')


    return render_template('card.html', card=card, related_cards=related_cards[start:end], page=page, total_pages=total_pages)

def cosine_similarity(embedding1, embedding2):
    return spatial.distance.cosine(embedding1, embedding2)

def fetch_related_cards(sfid):
    print(f' Fetching related cards for {sfid}...')
    
    embedding_ref = embeddings_similar[sfid]

    # Find the card with the highest cosine similarity to the embedding_ref
    embeddings = embeddings_similar.values()

    dists = [cosine_similarity(embedding_ref, embedding) for embedding in embeddings]

    # Zip the SFIDs and distances together
    sfids_and_dists = zip(embeddings_similar.keys(), dists)

    # Sort the SFIDs and distances by distance
    sfids_and_dists = sorted(sfids_and_dists, key=lambda x: x[1])
    
    top_k = 100
    related_sfids_and_dists = sfids_and_dists[:top_k]

    #print(f' Related SFIDs: {related_sfids}')
    related_cards = []
    for sfid, dist in related_sfids_and_dists:
        if sfid in cards_by_sfid:
            card_copy = json.loads(json.dumps(cards_by_sfid[sfid]))
            card_copy['distance'] = dist
            
            related_cards.append(card_copy)
        else:
            print(f'  {sfid} not found in cards_by_sfid')

    return related_cards

if __name__ == '__main__':
    app.run(debug=True)

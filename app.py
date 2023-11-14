from flask import Flask, render_template, request, jsonify
import chromadb
import json

app = Flask(__name__)

# Load Chroma database
db_dir = 'data/oracle-cards_db'

client = chromadb.PersistentClient(path=db_dir)
collection_names = client.list_collections()
# Load collections
collections = {}
for collection_name in collection_names:
    print(f'Loading collection name: "{collection_name}"')
#    collection = client.get_collection(collection_name)
#    collections[collection_name] = collection

default_collection = client.get_collection('similar') #, metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 1000, "hnsw:search_ef": 200})

cached_ids = default_collection.get(include=[])['ids']

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
        elif card['set_type'] in ['memorabilia', 'token', 'minigame']:
            card_data.remove(card)
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
            else:
                print(f'  {card["name"]} ({card["id"]}) has no image_uris')
                card_data.remove(card)

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

    related_cards, related_distances = fetch_related_cards(sfid)  # Adjust this function to return paginated results
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = len(related_cards) // per_page + (1 if len(related_cards) % per_page else 0)

    #print(f'Related cards: {related_cards[start:end]}')
    for related_card in related_cards[start:end]:
        if not 'image_uris' in related_card:
            print(f'  {related_card["name"]} ({related_card["id"]}) has no image_uris')


    return render_template('card.html', card=card, related_cards=related_cards[start:end], related_distances=related_distances[start:end], page=page, total_pages=total_pages)

def fetch_related_cards(sfid):
    print(f' Fetching related cards for {sfid}...')
    result = default_collection.get(ids=[sfid], include=["embeddings"])
    embeddings = result['embeddings'][0]
    #print(f' Embeddings: {embeddings}')

    # Grab related SFIDs from the collection with their respective distances.
    response = default_collection.query(
        query_embeddings=embeddings,
        n_results=2000,
    )

    related_sfids = response['ids'][0]
    related_distances = response['distances'][0]

    #print(f' Related SFIDs: {related_sfids}')
    related_cards = []
    for sfid in related_sfids:
        if sfid in cards_by_sfid:
            related_cards.append(cards_by_sfid[sfid])
        else:
            print(f'  {sfid} not found in cards_by_sfid')

    return related_cards, related_distances

if __name__ == '__main__':
    app.run(debug=True)

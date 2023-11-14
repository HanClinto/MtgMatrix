import json
import sys
import argparse
import pickle
from datetime import datetime

from tqdm import tqdm

import chromadb

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings

# NOTE: Leave 'name' out of the similarity calculations.
all_fields = ['type_line', 'mana_cost', 'cmc', 'power', 'toughness', 'loyalty', 'color_identity', 'produced_mana', 'oracle_text', 'flavor']

query_instructions = [
    ('similar', 'Represent the Magic: The Gathering card for retrieving similar cards: ',
        all_fields),
    ('duplicate', 'Represent the Magic: The Gathering card for retrieving duplicate cards: ',
        ['type_line', 'mana_cost', 'color_identity', 'produced_mana', 'oracle_text']),
    ('dupe2', 'Represent the Magic: The Gathering card for retrieving duplicate cards: ',
        ['type_line', 'mana_cost', 'oracle_text']),
    ('spike', 'Represent the functionality of the Magic: The Gathering card in terms of its overall effect on the game for retrieval of similar cards: ',
        ['type_line', 'cmc', 'power', 'produced_mana', 'oracle_text']),
    ('melvin', 'Represent the themes of the Magic: The Gathering card in terms of creature types, deck archetypes, and other functional archetypes (such as counters, planeswalkers, aristocrats, infect, etc) for retrieval of related cards: ',
        ['type_line', 'produced_mana', 'oracle_text']),
    ('vorthos', 'Represent the flavor of the Magic: The Gathering card in terms of its themes, characters, emotions, and flavor text for retrieval of related cards: ',
        ['name', 'type_line', 'oracle_text', 'color_identity', 'flavor']),
    ('timmy', 'Represent the power of the Magic: The Gathering card in terms of its mana cost, power, toughness, efficiency, and other numerical values for retrieval of similar cards: ',
        ['type_line', 'mana_cost', 'cmc', 'color_identity', 'power', 'toughness', 'produced_mana', 'oracle_text']),
    ('johnny', 'Represent the creativity of the Magic: The Gathering card in terms of its interactions with other cards, combos, complexity, and other creative uses for retrieval of related cards: ',
        ['produced_mana', 'oracle_text']),
]

input_file = 'oracle-cards-20231113220154.json'
output_dir = input_file.replace('.json', '_db')
model_name = 'hkunlp/instructor-large'

# Load card data
with open(input_file) as f:
    card_data = json.load(f)

    # Remove every card that has layout 'art_series' or 'token' or 'double_faced_token'
    # Remove every card that has a set_type of 'memorabilia' or 'token' or 'minigame'
    for card in card_data:
        if card['layout'] in ['art_series', 'token', 'double_faced_token']:
            card_data.remove(card)
        elif card['set_type'] in ['memorabilia', 'token', 'minigame']:
            card_data.remove(card)

    cards_by_sfid = {card['id']: card for card in card_data}

# Create a text splitter
splitter = CharacterTextSplitter()

# Add documents to the DB
query_instruction = query_instructions[0][1]

def card_field(card, field):
    field_name = field.replace('_', ' ').title()
    if field in card:
        return f'\n{field_name}: {card[field]}'
    else:
        if 'card_faces' in card:
            field_texts = []
            for face in card['card_faces']:
                if field in face:
                    field_name = field.replace('_', ' ').title()
                    field_texts.append(f'{field_name}: {face[field]}')
            if len(field_texts) > 0:
                return '\n' + ' // '.join(field_texts)
    return f'\n{field_name}: n/a'


# Create a client for all of the collections
client = chromadb.PersistentClient(path=output_dir)

FORCE_RECALCULATE = False

print(f'Generating embeddings for {len(documents)} cards...')

for query_tag, embed_instruction, embed_fields in query_instructions:
    # Create an embeddings model for this query instruction
    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=embed_instruction)

    # Create a Chroma collection for this query tag
    tag_collection = client.get_or_create_collection(
        name=query_tag,
        #embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 1000, "hnsw:search_ef": 200})

    cached_embeddings = tag_collection.get(include=[])
    cached_ids = cached_embeddings['ids']

    print(f' Generating embeddings for {len(card_data)} cards ({len(cached_ids)} precached) for {query_tag}...')

    idx = 0
    # Generate embeddings for each document and add each document to the DB
    for card in tqdm(card_data):
        idx += 1

        card_id = card['id']

        # Replace all instances of the card name with 'this card'
        oracle_text = card_field(card, 'oracle_text')
        card_names = card['name'].split(' // ')
        for card_name in card_names:
            card_name = card_name.strip()
            oracle_text = oracle_text.replace(card_name, 'this card')
        card['oracle_text'] = oracle_text.replace('\noracle_text: ', '')

        card_content = ''
        # Build a text representation of each card
        # NOTE: Each representation includes a different set of fields, because each axis cares about different things.
        # NOTE: We are not including the name in any of the representations (except for Vorthos), because the name is not a good indicator of similarity.
        for field in embed_fields:
            card_content += card_field(card, field)

        # Check to see if this ID is already in the collection
        if card_id in cached_ids and not FORCE_RECALCULATE:
            continue

        embeddings = embedding_function.embed_documents([card_content])

        tag_collection.upsert(embeddings=embeddings, ids=[card_id])

    print(f' {idx} cards added to the DB for {query_tag}.')
    
    cached_embeddings = tag_collection.get(include=['embeddings'])
    cached_ids = cached_embeddings['ids']

    embeddings_by_id = {}
    for idx, card_id in enumerate(cached_ids):
        embeddings_by_id[card_id] = cached_embeddings['embeddings'][idx]

    print(f' Writing {len(cached_ids)} embeddings to embeddings_{output_dir}_{query_tag}.pickle...')
    # Output the embeddings to pickle for easier debugging and re-use in other indices
    with open(f'embeddings_{output_dir}_{query_tag}.pickle', 'wb') as f:
        pickle.dump(embeddings_by_id, f)
    #with open(f'embeddings_{output_dir}_{query_tag}.json', 'w') as f:
    #    json.dump(embeddings_by_id, f, indent=0)
    print(f'  Done writing embeddings!')

# TODO: Combine all of the DBs into a single DB.
# db = Chroma(persist_directory=output_dir)



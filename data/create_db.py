import json
import sys
import argparse
import pickle
from datetime import datetime

from tqdm import tqdm

from scipy import spatial
import numpy as np

import faiss

from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings

# NOTE: Leave 'name' out of the similarity calculations.
all_fields = ['type_line', 'mana_cost', 'cmc', 'power', 'toughness', 'loyalty', 'color_identity', 'produced_mana', 'oracle_text', 'flavor_text']

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
        ['oracle_text', 'color_identity', 'flavor_text']),
    ('timmy', 'Represent the power of the Magic: The Gathering card in terms of its mana cost, power, toughness, efficiency, and other numerical values for retrieval of similar cards: ',
        ['type_line', 'mana_cost', 'cmc', 'color_identity', 'power', 'toughness', 'produced_mana', 'oracle_text']),
    ('johnny', 'Represent the creativity of the Magic: The Gathering card in terms of its interactions with other cards, combos, complexity, and other creative uses for retrieval of related cards: ',
        ['produced_mana', 'type_line', 'oracle_text']),
    ('flavor', 'Represent the flavor of the Magic: The Gathering card in terms of its themes, characters, emotions, and flavor text for retrieval of related cards: ',
        ['flavor_text'])
]

# For debug, only use the first query instruction
#query_instructions = [query_instructions[-1]]

input_file = 'oracle-cards-20231113220154.json'
output_dir = input_file.replace('.json', '_db')
model_name = 'hkunlp/instructor-large'

# Load card data
with open(input_file) as f:
    card_data = json.load(f)
    # Remove every card that has layout 'art_series' or 'token' or 'double_faced_token'
    # Remove every card that has a set_type of 'memorabilia' or 'token' or 'minigame'
#    for card in card_data:
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
                
    cards_by_sfid = {card['id']: card for card in card_data}

def get_card_fields(card, field):
    fields = []

    if field in card:
        fields.append(str(card[field]))
    else:
        if 'card_faces' in card:
            for face in card['card_faces']:
                if field in face:
                    fields.append(str(face[field]))

    # Trim and remove empty fields
    fields = [field.strip() for field in fields]
    fields = [field for field in fields if len(field) > 0]

    return fields

def get_readable_card_fields(card, field):
    field_name = field.replace('_', ' ').title()

    fields = get_card_fields(card, field)

    if len(fields) > 0:
        return f'\n{field_name}: ' + ' // '.join(fields)
    else:
        return f'\n{field_name}: n/a'

def cosine_similarity(embedding1, embedding2):
    return spatial.distance.cosine(embedding1, embedding2)

def dot_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)

FORCE_RECALCULATE = False

for query_tag, embed_instruction, embed_fields in query_instructions:

    cached_embeddings = {}

    # Load the simple cache from the Pickle file for each query tag
    try:
        with open(f'embeddings_{output_dir}_{query_tag}.pickle', 'rb') as f:
            cached_embeddings = pickle.load(f)
    except:
        pass

    cached_ids = cached_embeddings.keys()
    print(f' Loaded {len(cached_ids)} cached embeddings for {query_tag}.')
    #print(f'  Embedding length: {len(list(cached_embeddings.values())[0])}')
    embedding_size = 768

    # Create an embeddings model for this query instruction
    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=embed_instruction)

    print(f' Generating embeddings for {len(card_data)} cards ({len(cached_ids)} precached) for {query_tag}: [{str(embed_fields)}]...')

    idx = 0
    # Generate embeddings for each document and add each document to the DB
    for card in tqdm(card_data):
        idx += 1
        card_id = card['id']

        card_content = ''
        num_valid_fields = 0

        # Build a text representation of each card
        # NOTE: Each representation includes a different set of fields, because each axis cares about different things.
        # NOTE: We are not including the name in any of the representations (except for Vorthos), because the name is not a good indicator of similarity.
        for field in embed_fields:
            num_valid_fields += len(get_card_fields(card, field))
            card_content += get_readable_card_fields(card, field)

        # If the card fields are empty, skip this card.
        if num_valid_fields == 0:
            print(f'  Skipping card {card_id}: {card["name"]} because it has no fields.')
            # Ensure that it's removed from the cache and the database
            if card_id in cached_ids:
                del cached_embeddings[card_id]
            continue

        card_content = card_content.strip()

        #print(f'  Generating embedding for card {card_id}: {card["name"]}...')
        #print(f'   {card_content}')
        #print(f'   ({num_valid_fields} valid fields)')
        #exit()

        # Check to see if we have a cached embedding for this card
        if card_id in cached_ids and not FORCE_RECALCULATE:
            embeddings = cached_embeddings[card_id]
        else:
            # Replace all instances of the card name with 'this card' in the card content
            card_names = card['name'].split(' // ')
            for card_name in card_names:
                card_name = card_name.strip()
                card_content = card_content.replace(card_name, 'this card')

            # Calculate the card content
            embeddings = embedding_function.embed_documents([card_content])

        # Add it to the cache
        cached_embeddings[card_id] = np.array(embeddings)

        # Assert that the embeddings are the correct size
        assert len(embeddings) == embedding_size

    # Create a FAISS index for this query tag of type "IDMap,Flat"
    tag_collection = faiss.IndexFlatIP(embedding_size)
        #faiss.IndexIDMap(
        #)

    # Print stats about the index
    print(f'  Index stats for {query_tag}:')
    print(f'   is_trained: {tag_collection.is_trained}')
    print(f'   ntotal: {tag_collection.ntotal}')

    embeddings = list(cached_embeddings.values())
    keys = list(cached_embeddings.keys())
    keys_by_index = {idx: key for idx, key in enumerate(keys)}
    indices_by_key = {key: idx for idx, key in enumerate(keys)}

    print(f' Embeddings have len {len(embeddings)} and keys have shape {len(keys)}.')

    # Add all of the embeddings to the index
    embeddings = np.array(embeddings)
    keys = np.array(keys)
    print(f' Embeddings have len {len(embeddings)} and keys have shape {len(keys)}.')
    print(f' Embeddings have shape {embeddings.shape} and keys have shape {keys.shape}.')
    #tag_collection.add_with_ids(embeddings, keys)
    tag_collection.add(embeddings)


    print(f' {idx} cards added to the DB for {query_tag}.')

    print(f'  Index stats for {query_tag}:')
    print(f'   is_trained: {tag_collection.is_trained}')
    print(f'   ntotal: {tag_collection.ntotal}')

    
    print(f' Writing {len(cached_ids)} embeddings to embeddings_{output_dir}_{query_tag}.pickle...')

    # Output the embeddings to pickle for easier debugging and re-use in other indices
    with open(f'embeddings_{output_dir}_{query_tag}.pickle', 'wb') as f:
        pickle.dump(cached_embeddings, f)
    #with open(f'embeddings_{output_dir}_{query_tag}.json', 'w') as f:
    #    json.dump(cached_embeddings, f, indent=0)
    print(f'  Done writing embeddings for {query_tag}!')

    # Save tag_collection to disk
    print(f' Writing index to faiss_{output_dir}_{query_tag}.index...')
    faiss.write_index(tag_collection, f'faiss_{output_dir}_{query_tag}.index')

    # Build a mapping of keys to indices, and save it to disk alongside the FAISS index.
    print(f' Writing keys to faiss_{output_dir}_{query_tag}.keys...')
    #keys_by_index = {idx: key for idx, key in enumerate(keys)}
    indices_by_keys = {key: idx for idx, key in enumerate(keys)}

    with open(f'faiss_{output_dir}_{query_tag}.keys', 'w') as f:
        json.dump(indices_by_keys, f, indent=0)

    continue

    # Test the collection by ensuring that a brute-force search returns the same results as the HNSW search
    print(f' Testing {query_tag} collection...')

    # Get the first 10 cards
    test_cards = card_data[:5]

    error_count = 0

    embeddings = cached_embeddings.values()
        

    for test_card in test_cards:
        test_card_id = test_card['id']
        top_k = 4

        print(f'  Testing {test_card_id}...')

        # Get the embeddings for this card
        test_card_embeddings = cached_embeddings[test_card_id]

        then = datetime.now()

        # Search for the 10 most similar cards using FAISS
        faiss_dists, faiss_sfids = tag_collection.search(np.array([test_card_embeddings]), top_k)
        faiss_dists = faiss_dists[0]
        faiss_sfids = [keys_by_index[idx] for idx in faiss_sfids[0]]

        print(f'  FAISS search took {datetime.now() - then}.')

        # Search for the 10 most similar cards using brute force
        then = datetime.now()

        #bf_dists = [cosine_similarity(test_card_embeddings, embedding) for embedding in embeddings]
        bf_dists = [dot_similarity(test_card_embeddings, embedding) for embedding in embeddings]

        # Zip the SFIDs and distances together
        sfids_and_dists = zip(cached_embeddings.keys(), bf_dists)

        # Sort the SFIDs and distances by distance
        #sfids_and_dists = sorted(sfids_and_dists, key=lambda x: x[1])
        sfids_and_dists = sorted(sfids_and_dists, key=lambda x: x[1], reverse=True)
        
        bf_sfids = [sfid for sfid, dist in sfids_and_dists[:top_k]]
        bf_dists = [dist for sfid, dist in sfids_and_dists[:top_k]]

        print(f'  Brute force search took {datetime.now() - then}.')

        print(f' Search results for {test_card_id}:')
        print(f'  FAISS results:')
        for faiss_sfid, faiss_dist in zip(faiss_sfids, faiss_dists):
            print(f'   {faiss_dist}: {faiss_sfid}')

        print(f'  Brute force results:')
        for bf_sfid, bf_dist in sfids_and_dists[:top_k]:
            print(f'   {bf_dist}: {bf_sfid}')

        for bf_sfid in bf_sfids:
            if bf_sfid not in faiss_sfids:
                print(f'  ERROR: {bf_sfid} not in {faiss_sfids}')
                error_count += 1

    print(f' {error_count} errors found in {query_tag} collection.')



        


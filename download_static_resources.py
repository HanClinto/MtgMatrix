# Download content from https://api.scryfall.com/symbology and store symbol SVGs in the appropriate folder in static/symbols

import requests
import json
import os
import shutil

# Create the ./static/symbols folder if it doesn't exist
if not os.path.exists('./static/symbols'):
    os.mkdir('./static/symbols')

# Get the list of symbols
r = requests.get('https://api.scryfall.com/symbology')

# Load the list of symbols
symbols = json.loads(r.content)['data']

symbol_dict = {}

# For each symbol, download the svg_uri and save it to the appropriate folder
for symbol in symbols:
    # Get the symbol name
    symbol_name = symbol['symbol']
    # Get the symbol SVG URI
    symbol_uri = symbol['svg_uri']
    # Get the symbol SVG
    r = requests.get(symbol_uri)

    # Get the clean name from the symbol_uri
    clean_name = symbol_uri.split('/')[-1]

    # Save the symbol SVG to the appropriate folder
    with open(f'static/symbols/{clean_name}', 'wb') as f:
        f.write(r.content)
    print(f'Downloaded symbol "{symbol_name}" to static/symbols/{clean_name}')

    # Add the symbol to the symbol_dict
    symbol_dict[symbol_name] = clean_name

# Save the symbol_dict to static/symbols.json
with open('static/symbols.json', 'w') as f:
    json.dump(symbol_dict, f, indent=2)

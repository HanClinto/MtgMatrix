# MtgMatrix
Use semantic search to browse similar and related cards in Magic: The Gathering. Powered by Langchain.


## App Description

### Back-end
* Downloads all Magic: The Gathering card information from Scryfall.com via bulk data download
* Formats the relevant data for each card into a text string suitable for processing via LLM
* Uses Langchain and INSTRUCTOR to generate semantic embeddings and store each card document in a local vector database


### Front-end

* Allows users to browse Magic: The Gathering cards and do simple search by name, colors, etc.
* For each card, displays similar cards in a ranked list.  Users can click on each of these cards to browse to that card and review similar cards with that as the starting point.


## TODO:

* [ ] Download card symbol images from https://api.scryfall.com/symbology and do text-wide substitution anywhere on the site.
* [ ] Allow users to type in theoretical Magic: The Gathering cards and let the system display related cards that are most similar to it, cards that would combo / synergize with it, etc.
* [ ] Embed information from each card from several different vantage-points, such as card functionality ("Spike"), card lore / flavor-text ("Vorthos"), card complexity, combos, and rules-interactions ("Johnny"), creature type and size / efficiency ("Timmy"), etc.  This will let us find the axis on how each card is related.
# python scripts/chunk_data.py 

import os
import re
import json

def split_by_sec_items(text):
    """
    Split a 10-K document into items based on standard SEC headers.
    Returns a dictionary mapping Item name to its text content.
    """
    # Regex to capture "ITEM 1. BUSINESS", etc. 
    # Must be at the start of a line and followed by a title or at least some text on the same line.
    # In the index, these are often just "Item 1." followed immediately by a newline or page number.
    item_pattern = re.compile(r'^\s*(ITEM\s+[0-9A-C]+\.?\s+[A-Z].+)\s*$', re.MULTILINE)
    
    # We also want to capture "PART X" as headers if they are substantial
    part_pattern = re.compile(r'^\s*(PART\s+[I-V]+)\s*$', re.MULTILINE)
    
    # Find all matches
    matches = list(item_pattern.finditer(text))
    
    # If no ITEM matches with titles, try broader search but we'll use the 200 char rule later
    if not matches:
        item_pattern = re.compile(r'^\s*(ITEM\s+[0-9A-C]+\.?[^\n]+)\n', re.IGNORECASE)
        matches = list(item_pattern.finditer(text))
    
    if not matches:
        return {"Full Document": text}
    
    sections = {}
    
    # Preliminary pass: identify where the REAL content starts.
    # Usually the first ITEM 1. BUSINESS with a lot of text after it.
    start_pos_actual = 0
    for match in matches:
        if "ITEM 1." in match.group(1).upper() and "BUSINESS" in match.group(1).upper():
            start_pos_actual = match.start()
            break
            
    # We'll treat everything before the real Item 1 as "Front Matter"
    # but we skip most of it if it's just the index.
    sections["Front Matter"] = text[:start_pos_actual].strip()
    
    last_pos = start_pos_actual
    last_item_name = "Front Matter"
    
    # Filter matches to only those after or at start_pos_actual
    actual_matches = [m for m in matches if m.start() >= start_pos_actual]
    
    for match in actual_matches:
        start_pos = match.start()
        # The content of the previous section
        if last_item_name:
            sections[last_item_name] = text[last_pos:start_pos].strip()
        
        # New section starts
        last_item_name = match.group(1).strip().upper()
        last_pos = match.end()
        
    # Last section
    sections[last_item_name] = text[last_pos:].strip()
    
    return sections

def sentence_aware_chunker(text, chunk_size=2000, overlap_sentences=2):
    """
    Splits text into chunks based on sentence boundaries.
    """
    # Simple sentence splitter (matches . ! ? followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we already have some content
        if current_length + sentence_len > chunk_size and current_chunk:
            # Join the current chunk and add it
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            # We take the last 'overlap_sentences' from the current chunk
            overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else current_chunk
            current_chunk = list(overlap)
            current_chunk.append(sentence)
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_length > 0 else 0)
            
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    input_file = "data/tesla/2025.txt"
    output_file = "data/tesla/chunks.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Splitting into SEC Items...")
    items = split_by_sec_items(text)
    
    all_chunks = []
    global_index = 0
    
    for item_name, item_content in items.items():
        # Skip sections that are too small (likely index entries or noise)
        if len(item_content) < 200:
            continue
            
        item_chunks = sentence_aware_chunker(item_content)
        
        for i, chunk in enumerate(item_chunks):
            word_count = len(chunk.split())
            all_chunks.append({
                "id": f"chunk_{global_index}",
                "source": item_name,
                "content": chunk,
                "chunk_index": global_index,
                "item_index": i,
                "word_count": word_count
            })
            global_index += 1
            
    print(f"Total chunks created: {len(all_chunks)}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print(f"Successfully saved to {output_file}")

if __name__ == "__main__":
    main()

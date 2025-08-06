from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size=200, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

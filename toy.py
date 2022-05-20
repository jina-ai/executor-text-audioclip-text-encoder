from executor.audioclip_text import AudioCLIPTextEncoder
from jina import Document, DocumentArray, Executor

def main():
    docs = DocumentArray([Document(id='root1', text=text)])
    docs[0].chunks = [
        Document(id='chunk11', text=text),
        Document(id='chunk12', text=text),
        Document(id='chunk13', text=text),
    ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]


if __name__=="__main__":
    main()
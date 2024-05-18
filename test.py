
import urllib
import urllib.request

def research_paper(topic: str):
    import urllib, urllib.request
    url = 'http://export.arxiv.org/api/query?search_query=all:machine&start=0&max_results=1'
    data = urllib.request.urlopen(url)
    print(data.read().decode('utf-8'))

    return data.read().decode('utf-8')

print(research_paper("LLM agent"))
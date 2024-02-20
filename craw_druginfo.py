import requests
import time
# import urllib3
# urllib3.disable_warnings()

headers={
    'Referer':'https://go.drugbank.com/releases/latest/', 'Connection': 'close',
}
requests.adapters.DEFAULT_RETRIES = 100


with open('./data/DrugComb/drug_id.csv', 'r') as f:
    for s in f:
        ss = requests.session()
        ss.keep_alive = False
        # ss.proxies = {"https": "57.10.114.47:8000", "http": "32.218.1.7:9999", }
        a,b = s.split(',')
        response = ss.get(f"https://go.drugbank.com/structures/small_molecule_drugs/{a}.pdb",headers= headers, verify=False)
        c = str(response.content, encoding = "utf-8")
        with open(f'druginfo/{a}.pdb', 'w') as wfile:
            wfile.writelines(c)

        
 

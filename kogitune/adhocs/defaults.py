DEFAULTS =[
    {
        "name": "mejiro",
        "date": "2023-12-24",
        "tokenizer_path": "",
        "max_tokens": 128,
    }
]

def search_list(at_version):
    if at_version is None:
        return DEFAULTS
    defaults = []
    found = None
    for defs in DEFAULTS:
        if defs['name'] == at_version:
            found = defs['date']
        if found and found <= defs['date']:
            defaults.append(defs)
    return defaults

def get_default(key:str, at_version:str):
    for defs in search_list(at_version):
        if key in defs:
            return defs[key]
    return None

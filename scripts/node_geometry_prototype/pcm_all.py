import requests, csv
SVC="https://pricecontourmap.spp.org/arcgis/rest/services/PCM/RTBM_Features/MapServer"
def pull(lid):
    q=requests.get(f"{SVC}/{lid}/query", params={"where":"1=1","outFields":"SETTLEMENT_LOCATION,PNODETYPE,DESCRIPTION",
        "returnGeometry":"true","outSR":"4326","f":"json"}, timeout=90).json()
    out={}
    for f in q.get("features",[]):
        g=f.get("geometry") or {}; a=f["attributes"]
        sl=a.get("SETTLEMENT_LOCATION")
        if sl and g.get("x") is not None: out[sl]=(round(g["y"],5), round(g["x"],5), a.get("PNODETYPE"))
    return out
pts={}; 
for lid,nm in [(1,"DCTie"),(2,"Hub"),(3,"Interface")]:
    d=pull(lid); pts.update({k:(*v,nm) for k,v in d.items()})
    print(f"layer {lid} {nm}: {len(d)}")
    if nm=="Interface": print("   ALL interfaces:", sorted(d.keys()))
print(f"TOTAL PCM points: {len(pts)}\n")

# join to 64 hub/BA list
rows=list(csv.DictReader(open("/tmp/claude-1000/-home-justinfields-Documents-github-spp-weis-price-forecast/fd760840-e901-4274-bfca-eabe18c3749f/scratchpad/west_hub_nodes.csv")))
out=[]; matched=0
for r in rows:
    p=pts.get(r["settlement_location"])
    if p: matched+=1
    out.append({**r,"lat":p[0] if p else "","lon":p[1] if p else "","pcm_layer":p[3] if p else ""})
print(f"64 hub/BA nodes matched to PCM coords: {matched}/{len(rows)}")
print("UNMATCHED:", [o['settlement_location'] for o in out if not o['lat']])

# also: coverage of full 302 SWPW list
allrows=list(csv.DictReader(open("/tmp/claude-1000/-home-justinfields-Documents-github-spp-weis-price-forecast/fd760840-e901-4274-bfca-eabe18c3749f/scratchpad/swpw_nodes_classified.csv")))
allm=sum(1 for r in allrows if r["settlement_location"] in pts)
print(f"full 302 SWPW nodes matched to PCM coords: {allm}/{len(allrows)}")

csv.DictWriter(open("/tmp/claude-1000/-home-justinfields-Documents-github-spp-weis-price-forecast/fd760840-e901-4274-bfca-eabe18c3749f/scratchpad/west_hub_nodes_latlon.csv","w"),
               fieldnames=list(out[0].keys())).writerows([]) or None
w=csv.DictWriter(open("/tmp/claude-1000/-home-justinfields-Documents-github-spp-weis-price-forecast/fd760840-e901-4274-bfca-eabe18c3749f/scratchpad/west_hub_nodes_latlon.csv","w"),fieldnames=list(out[0].keys()))
w.writeheader(); w.writerows(out)
print("\nMATCHED hub/BA nodes (name, class, lat, lon, layer):")
for o in out:
    if o['lat']: print(f"  {o['settlement_location']:20} {o['class']:4} {o['scope']:14} {o['lat']:>9}, {o['lon']:>10}  [{o['pcm_layer']}]")

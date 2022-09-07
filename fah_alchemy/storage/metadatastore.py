from typing import Dict, List
import weakref

from gufe import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable
from gufe.storage.metadatastore import MetadataStore
from py2neo import Graph, Node, Relationship, Subgraph


class Neo4jStore:

    def __init__(self, graph: "py2neo.Graph"):
        self.graph = graph
        self.gufe_nodes = weakref.WeakValueDictionary()
    
    def _subgraph_from_gufe(self, sdct: Dict, labels: List[str], gufe_key, org, campaign, project):
        sdct['_gufe_key'] = str(gufe_key)
        sdct.update({'_org': org, '_campaign': campaign, '_project': project})
        sdct['_scoped_key'] = [sdct['_gufe_key'], org, campaign, project]
        
        subgraph = Subgraph()
        node = Node(*labels)
        for key, value in sdct.items():
            if isinstance(value, dict):
                if all([isinstance(x, GufeTokenizable) for x in value.values()]):
                    for k, v in value.items():
                        node_ = subgraph_ = self.gufe_nodes.get((v.key, org, campaign, project))
                        if node_ is None:
                            subgraph_, node_ = self._subgraph_from_gufe(v.to_shallow_dict(), labels=['GufeTokenizable', v.__class__.__name__], gufe_key=v.key, org=org, campaign=campaign, project=project)
                            self.gufe_nodes[(str(v.key), org, campaign, project)] = node_
                        subgraph = subgraph | Relationship.type("DEPENDS_ON")(node, node_, attribute=key, key=k, _org=org, _campaign=campaign, _project=project) | subgraph_    
                else:
                    node[key] = "__dict__: " + str(value)
            elif isinstance(value, list):
                # lists can only be made of a single, primitive data type
                # we encode these as strings with a special starting indicator
                if isinstance(value[0], (int, float, str)) and all([isinstance(x, type(value[0])) for x in value]):
                    node[key] = value
                elif all([isinstance(x, GufeTokenizable) for x in value]):
                    for i, x in enumerate(value):
                        node_ = subgraph_ = self.gufe_nodes.get((x.key, org, campaign, project))
                        if node_ is None:
                            subgraph_, node_ = self._subgraph_from_gufe(x.to_shallow_dict(), labels=['GufeTokenizable', x.__class__.__name__], gufe_key=x.key, org=org, campaign=campaign, project=project)
                            self.gufe_nodes[(x.key, org, campaign, project)] = node_
                        subgraph = subgraph | Relationship.type("DEPENDS_ON")(node, node_, attribute=key, index=i, _org=org, _campaign=campaign, _project=project) | subgraph_
                else:
                    node[key] = "__list__: " + str(value)
            elif isinstance(value, tuple):
                # lists can only be made of a single, primitive data type
                # we encode these as strings with a special starting indicator
                if not (isinstance(value[0], (int, float, str)) and all([isinstance(x, type(value[0])) for x in value])):
                    node[key] = "__tuple__: " + str(value)
            elif isinstance(value, GufeTokenizable):
                node_ = subgraph_ = self.gufe_nodes.get((value.key, org, campaign, project)) 
                if node_ is None:
                    subgraph_, node_ = self._subgraph_from_gufe(value.to_shallow_dict(), labels=['GufeTokenizable', value.__class__.__name__], gufe_key=value.key, org=org, campaign=campaign, project=project)
                    self.gufe_nodes[(value.key, org, campaign, project)] = node_
                subgraph = subgraph | Relationship.type("DEPENDS_ON")(node, node_, attribute=key, _org=org, _campaign=campaign, _project=project) | subgraph_
            else:
                node[key] = value

        subgraph = subgraph | node

        return subgraph, node
        
    def create_network(self, network: AlchemicalNetwork, org, campaign, project):
        g, n = self._subgraph_from_gufe(network.to_shallow_dict(), 
                                    labels=['GufeTokenizable', network.__class__.__name__], 
                                    gufe_key=network.key, org=org, campaign=campaign, project=project)
        self.graph.create(g)
        
    def update_network(self, network: AlchemicalNetwork, org, campaign, project):
        
        ndict = network.to_shallow_dict()
        
        g, n = self._subgraph_from_gufe(ndict, 
                                    labels=['GufeTokenizable', network.__class__.__name__], 
                                    gufe_key=network.key, org=org, campaign=campaign, project=project)
        self.graph.merge(g, 'GufeTokenizable', '_scoped_key')
        
    def retrieve(key: "gufe.GufeKey"):
        ...

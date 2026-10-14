# Pocket--ai-trader

## Développement sans proxy — vérification du 14 octobre 2026

Configuration sûre à conserver dans les variables d'environnement du service :

```env
PO_PROXY_ENABLED=false
TRADING_MODE=demo
TRADING_ENABLED=false
PO_LIVE_ENABLED=false
```

Aucun hôte, port, utilisateur ou mot de passe de proxy n'est requis.
`PO_PROXY_ENABLED` vaut False par défaut dans `Config.py` et
`po_connector.py`. Il est également visible dans `get_status()` du connecteur.
Ce drapeau est déclaratif : aucun transport proxy n'existe dans ce dépôt.
Le client externe injecté reste responsable de sa connexion. Activer ce drapeau
seul ne configure donc pas de proxy et ne change pas un client déjà injecté.

`server.py` ne charge pas `Config.py` et ne charge pas automatiquement `.env`.
Le connecteur lit directement l'environnement du processus. Sur un hébergement,
renseigner les quatre variables ci-dessus dans la configuration du service.
`.env.example` est un modèle documentaire, pas une configuration chargée
automatiquement par tous les modules.

### Changements et justification

- `Config.py` : défaut `PO_PROXY_ENABLED=False`.
- `po_connector.py` : même défaut et indication dans le statut ; aucune méthode
  de connexion ou d'exécution supprimée ou modifiée.
- `.env.example` : quatre valeurs sûres, aucun secret ni adresse fictive.
- `server.py` : uniquement désindentation de la déclaration
  `calculate_position_size`, qui provoquait une `IndentationError` à la ligne
  1247 dans la version originale. Correction nécessaire au démarrage.
- `tests/test_no_proxy.py` : tests avec option absente puis explicitement false,
  sans identifiants ni client externe ; vérification des verrous et HTTP 200
  sur `/`, `/health`, `/config` via TestClient.

### Résultats des tests dans le sandbox Linux

- 15 tests réussis : 2 nouveaux tests sans proxy et 13 tests du stockage.
- Compilation Python de `server.py`, `Config.py`, `po_connector.py` réussie.
- Démarrage réel `python server.py` réussi avec les quatre variables sûres,
  sans HTTP_PROXY, HTTPS_PROXY ni ALL_PROXY. Arrêt propre après cinq secondes.
- Suite complète bloquée à la collecte : `test_order_manager_connector.py`
  importe `order_manager`, absent du dépôt. Le fichier existant porte le nom
  `ordre _manager.py` et son interface doit être examinée avant correction.
  Aucun renommage ni contournement de ce problème n'a été effectué.
- Aucun ordre envoyé, aucune authentification Pocket Option tentée.
- `/health` ne prouve ni la connexion à Pocket Option ni la disponibilité
  de Twelve Data ; aucune clé Twelve Data n'a été fournie pour ces tests.
- En démo, `execute_order()` conserve sa simulation locale existante :
  `confirmed=True` n'est pas une confirmation d'ordre par Pocket Option.

### Prochaines étapes depuis Android

1. Ne pas remplir l'ancien formulaire de secrets proxy.
2. Télécharger l'archive corrigée livrée dans cette conversation. Elle ne
   contient ni `.env`, ni secrets, ni environnement virtuel, ni historique Git.
3. Pour mettre à jour GitHub depuis le navigateur : ouvrir le dépôt, puis
   remplacer `Config.py`, `po_connector.py`, `server.py` et `README.md` par les
   versions de l'archive ; ajouter `.env.example` et `tests/test_no_proxy.py`.
   L'archive contient également `modifications.patch` pour une application par
   Git sur un environnement de développement. Ne pas remplacer les autres
   fichiers et ne pas téléverser de `.env` privé.
4. Si le dépôt est déjà relié à Render, définir les quatre variables sûres dans
   Environment avant le prochain déploiement. Garder la commande de démarrage
   existante si elle lance `server.py` ou `uvicorn server:app`.
5. Une fois déployé, ouvrir `<URL-du-service>/health` depuis le téléphone :
   résultat attendu `status: healthy`. Ouvrir aussi `/config` ; ce chemin
   indique que le serveur n'exécute pas d'ordres, mais n'affiche pas les verrous
   du connecteur. Les vérifier dans les variables du service.
6. Corriger ensuite séparément l'incohérence du gestionnaire d'ordres et
   relancer toute la suite. La connexion Pocket Option reste une étape
   distincte, sans proxy pour l'instant et sans mode réel.

Les changements livrés sont locaux : rien n'a été poussé sur GitHub ou déployé
sur Render par cette intervention.
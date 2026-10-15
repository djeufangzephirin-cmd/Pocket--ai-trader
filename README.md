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
- `order_manager.py` : module reconstruit (voir la section suivante).
- `ordre _manager.py` → `order_manager_ancien.py` : renommage, trace conservée.
- `tests/test_order_manager_safety.py`, `tests/test_order_manager_integration.py` :
  nouveaux tests de garde-fous et de bout en bout, sans réseau.
- `pytest.ini` : `pythonpath = .` pour que `pytest` seul fonctionne depuis la
  racine du dépôt, sans passer par `python -m pytest`.

### Gestionnaire d'ordres reconstruit — 15 octobre 2026

`order_manager.py` n'existait pas dans le dépôt. Le fichier présent,
`ordre _manager.py`, portait un nom inimportable (une espace) **et** une
`IndentationError` à la ligne 277 : une déclaration `def execute_order()` avait
été insérée au milieu d'une autre méthode. Le module était donc inexploitable,
et toute la suite de tests bloquée à la collecte.

Le module a été réécrit contre l'interface réellement exposée par le dépôt, et
non contre une interface supposée :

| Appelant | Interface exigée |
| --- | --- |
| `tests/test_order_manager_connector.py` | `submit_order(asset, direction, amount, payout, expiration_seconds)` → dict avec `status` |
| `po_connector.PocketOptionConnector` | `execute_order(order)` → bool |
| `data_store.DataStore` | `save_order(order_record)`, `get_daily_stats(date)` |

L'ancienne version appelait `connector.place_order(...)`, méthode qui n'existe
pas dans `po_connector.py` : elle n'aurait jamais pu fonctionner.

Règles appliquées, dans cet ordre :

1. Arrêt d'urgence actif → refus.
2. `asset` absent → refus. La casse est **préservée** : les symboles Pocket
   Option du type `#TSLA_otc` seraient corrompus par une mise en majuscules.
3. `amount` ≤ 0, ou supérieur à `max_position_size` → refus.
4. `direction` hors `BUY` / `SELL` → refus.
5. `expiration_seconds` inférieur à 60 → refus.
6. `payout` inférieur à 80 % → refus, avec le taux de réussite requis affiché.
7. Perte journalière réalisée supérieure ou égale à `max_daily_loss` → refus.

Deux garanties supplémentaires :

- **Persistance avant exécution.** Chaque ordre autorisé est écrit en base au
  statut `SUBMITTED` *avant* l'appel au connecteur. Si cette écriture échoue,
  l'ordre n'est pas envoyé et la réponse porte le statut `FAILED`. Un ordre ne
  peut donc jamais exister côté plateforme sans trace locale préalable.
  `tests/test_order_manager_integration.py` le vérifie depuis l'intérieur du
  connecteur, en lisant la base au moment de l'exécution.
- **Échec fermé.** Toute ambiguïté (montant illisible, perte journalière
  illisible, connecteur muet ou en exception) produit un refus ou un échec,
  jamais un envoi silencieux.

Cycle de vie d'un ordre : `SUBMITTED` → `ACCEPTED` ou `FAILED` ; `REJECTED`
pour tout refus antérieur à une tentative d'exécution. Chaque tentative laisse
exactement une ligne en base, y compris les refus.

`order_manager_ancien.py` est conservé comme trace. Ce fichier reste
volontairement non importable : il ne doit pas être utilisé.

### Résultats des tests dans le sandbox Linux

- **129 tests réussis** : 2 tests sans proxy, 13 tests du stockage, 8 tests du
  contrat connecteur/gestionnaire, 84 tests de garde-fous, 22 tests
  d'intégration.
- Compilation Python de `server.py`, `Config.py`, `po_connector.py`,
  `data_store.py` et `order_manager.py` réussie.
- Démarrage réel `python server.py` réussi avec les quatre variables sûres,
  sans HTTP_PROXY, HTTPS_PROXY ni ALL_PROXY. Arrêt propre après cinq secondes.
- La suite n'est plus bloquée à la collecte : `order_manager.py` existe et
  satisfait l'interface attendue par `tests/test_order_manager_connector.py`.
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
6. Corriger l'incohérence du gestionnaire d'ordres — **fait le 15 octobre
   2026** : `order_manager.py` reconstruit, `ordre _manager.py` renommé
   `order_manager_ancien.py`, deux fichiers de tests ajoutés, suite complète à
   129 tests réussis. Pour relancer localement, depuis la racine du dépôt :

   ```bash
   pytest -q
   ```

   La connexion Pocket Option reste une étape distincte, sans proxy pour
   l'instant et sans mode réel.

Les changements du 14 octobre 2026 ont été livrés localement, par archive.
La correction du gestionnaire d'ordres du 15 octobre 2026 est, elle, poussée
sur la branche `main` de ce dépôt.
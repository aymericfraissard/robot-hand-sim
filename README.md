# robot-hand-sim — Main robotique pilotée par webcam

Votre main est filmée par la webcam et suivie en temps réel par **MediaPipe**. Une **main robotique 3D**
(dans le navigateur) reproduit les mouvements de vos 5 doigts et l'**inclinaison de votre poignet**.
Le même code pilote plus tard une **vraie main** (5 ou 6 servos + Arduino) : il suffit de
changer `mode` dans `config.yaml`.

```
Webcam ─► OpenCV ─► MediaPipe HandLandmarker ─► angles des doigts ─► lissage ─┬─► sim    : WebSocket ─► page 3D (Three.js)
                                                  (calibration JSON)          └─► serial : USB ─► Arduino ─► 5 servos
```

## Sommaire
1. [Installation](#1-installation)
2. [Lancement](#2-lancement)
3. [Raccourcis clavier](#3-raccourcis-clavier)
4. [Calibration](#4-calibration)
5. [Poignet](#5-poignet-inclinaison)
6. [Configuration](#6-configuration-configyaml)
7. [Passer au vrai matériel](#7-passer-au-vrai-matériel)
8. [Dépannage](#8-dépannage)
9. [Fonctionnement et structure](#9-fonctionnement-et-structure)

---

## 1. Installation

**Prérequis :** Windows 10/11, **Python 3.13** (testé avec 3.13.14 et mediapipe 1.1.0), une webcam
et environ 1 Go d'espace disque libre.

Dans un terminal PowerShell (VS Code : *Terminal → Nouveau terminal*), placé dans le dossier du projet :

```powershell
# 1. Créer l'environnement virtuel
python -m venv .venv

# 2. L'activer (si PowerShell refuse les scripts, voir « Dépannage »)
.\.venv\Scripts\Activate.ps1

# 3. Installer les dépendances (versions épinglées)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Vérifier que tout fonctionne
python -m pytest
```

> ⚠️ N'installez pas `opencv-python` en plus : mediapipe installe déjà `opencv-contrib-python`
> et les deux paquets entrent en conflit (`cv2` cassé).

Le modèle MediaPipe `hand_landmarker.task` (~8 Mo) est **téléchargé automatiquement** dans `models/`
au premier lancement.

Dans **VS Code** : ouvrez le dossier `robot-hand-sim`. L'interpréteur du `.venv` est sélectionné
automatiquement grâce à `.vscode/settings.json`. Si ce n'est pas le cas : `Ctrl+Shift+P` →
*Python: Select Interpreter* → `.venv`.

## 2. Lancement

```powershell
python main.py --mode sim      # simulation 3D dans le navigateur
python main.py --mode serial   # vraie main (Arduino)
python main.py --mode both     # les deux en même temps
python main.py --demo          # sans webcam : mouvements automatiques (teste le visualiseur seul)
```

Ou dans VS Code avec **F5**, en choisissant la configuration dans le menu *Exécuter et déboguer* :
*simulation (sim)*, *Arduino (serial)*, *sim + Arduino (both)*, *Démo sans webcam* ou *Tests unitaires*.

En mode `sim`, deux fenêtres s'ouvrent :
- la **fenêtre webcam**, avec le squelette de la main, la flexion et l'angle de chaque doigt, le FPS
  et l'état (main détectée ou non) ;
- un **onglet du navigateur** (`http://127.0.0.1:8000`) avec la main 3D. Faites glisser la souris
  pour tourner, utilisez la molette pour zoomer et le clic droit pour déplacer la vue. Une pastille
  indique l'état de la connexion WebSocket. La page se reconnecte toute seule si vous relancez le
  programme.

Autres options : `--camera 1` (choisir une autre webcam), `--no-browser` (ne pas ouvrir l'onglet),
`--config autre.yaml`.

## 3. Raccourcis clavier

Ils sont actifs quand la fenêtre webcam a le focus.

| Touche | Action |
|---|---|
| `q` ou `Échap` | Quitter |
| `c` | Lancer la calibration (appuyer de nouveau pour l'annuler) |
| `m` | Activer ou désactiver l'affichage en miroir |

## 4. Calibration

Sans calibration, des valeurs moyennes sont utilisées. Elles fonctionnent, mais chaque main est
différente : calibrer améliore beaucoup la précision, surtout pour le pouce.

1. Placez votre main bien visible devant la webcam, à 40–70 cm, **coude posé** sur la table.
2. Appuyez sur **`c`**.
3. **« OUVREZ la main, doigts en haut »** : écartez les doigts, main bien tendue, **paume face à la
   caméra et doigts vers le haut**. Cette pose sert aussi de **position neutre du poignet** (inclinaison
   à 0). Après 3 s, la capture dure 1 s (ne bougez plus).
4. **« FERMEZ le poing »** : poing fermé, pouce replié sur les doigts. Même principe.
5. Le message « Calibration enregistrée ! » s'affiche.

La calibration est sauvegardée dans `calibration.json` et **rechargée à chaque démarrage**. Si un
doigt n'a pas assez bougé entre les deux poses, la calibration est refusée et l'ancienne est conservée.
Pour revenir aux valeurs par défaut, supprimez `calibration.json`.

## 5. Poignet (inclinaison)

Un seul mouvement du poignet est reproduit : l'**inclinaison** (déviation radiale / cubitale). Il
s'agit de pencher la main vers le pouce ou vers l'auriculaire, paume face à la caméra. L'angle est
positif vers le pouce.

Ne sont **pas** reproduites :
- la **rotation** de la main sur elle-même (tourner comme une poignée de porte), retirée car trop
  instable avec une seule webcam ;
- la **flexion** (main vers le haut ou vers le bas), qui se lit dans la profondeur, que la webcam
  estime mal.

Tourner la main sur elle-même ne fait pas bouger l'inclinaison.

**Comment c'est mesuré.** MediaPipe ne voit que la main, pas l'avant-bras. On compare donc
l'orientation de la main à la **pose neutre** enregistrée pendant la calibration (main ouverte, doigts
vers le haut). Cette méthode suppose que **l'avant-bras reste à peu près immobile** : posez le coude
sur la table. Sans calibration, la pose neutre par défaut est « paume face à la caméra, doigts vers le
haut ».

**Bon à savoir**
- L'angle est limité à `range_deg` (±30° par défaut).
- Quand la main disparaît, le poignet revient doucement en position neutre.
- La main 3D est une **main droite**. Elle s'utilise naturellement avec votre main droite, vue en
  miroir. Avec la main gauche, le sens de l'inclinaison reste anatomique (« vers le pouce »).
- Pour désactiver le poignet : `wrist.enabled: false` (la trame série revient à 5 valeurs).

## 6. Configuration (`config.yaml`)

**Tous les réglages se trouvent dans `config.yaml`.** Les listes de 5 valeurs sont dans l'ordre
pouce, index, majeur, annulaire, auriculaire.

| Clé | Rôle |
|---|---|
| `mode` | `sim`, `serial` ou `both` (remplaçable par `--mode`) |
| `camera.index` / `width` / `height` / `fps` | Webcam utilisée et résolution |
| `camera.api` | `dshow` (Windows, recommandé), `msmf` ou `any` |
| `camera.mirror` | Affichage miroir au démarrage |
| `camera.exposure` | `-5` (défaut, garantit 30 FPS) ou `auto` (image plus claire, mais 15 FPS dans une pièce sombre) |
| `angles.smoothing_alpha` | Lissage : `1.0` = aucun lissage, `0.2` = très lisse mais avec du retard |
| `angles.return_alpha` | Vitesse du retour vers « main ouverte » quand la main disparaît |
| `angles.min_angles` / `max_angles` | Angle servo à 0 % (doigt tendu) et à 100 % (doigt plié), par doigt |
| `angles.invert` | `true` pour un servo monté à l'envers (inverse son sens) |
| `wrist.enabled` | Active l'inclinaison du poignet (trame série à 6 valeurs au lieu de 5) |
| `wrist.smoothing_alpha` | Lissage du poignet |
| `wrist.deviation.*` | `range_deg` (angle mesuré qui correspond à `min_angle` / `max_angle`), `min_angle`, `max_angle` (servo), `invert` |
| `calibration.*` | Fichier de calibration et durées du compte à rebours et de la capture |
| `sim.http_port` / `ws_port` | Ports du serveur web et du WebSocket |
| `serial.port` / `baudrate` | Port COM de l'Arduino et vitesse (115200) |
| `serial.send_rate_hz` | Fréquence maximale d'envoi des trames |

`invert`, `min_angles` / `max_angles` et `min_angle` / `max_angle` ne s'appliquent qu'aux **angles
servo**, c'est-à-dire aux valeurs envoyées à l'Arduino. La main 3D affiche toujours les mouvements
« logiques » : un servo inversé ne retourne donc pas le doigt (ou le poignet) dans la simulation.

## 7. Passer au vrai matériel

### Matériel

| Élément | Conseil |
|---|---|
| Microcontrôleur | **Arduino Uno** ou **Nano** (ou ESP32 avec la bibliothèque `ESP32Servo`) |
| Servos doigts | 5 × **MG90S** (petite main, ~1 A en pointe chacun) ou **MG996R** (main grandeur nature type InMoov, ~2,5 A en pointe chacun) |
| Servo poignet | 1 × **MG996R** (ou plus puissant) pour l'inclinaison |
| Alimentation servos | **5–6 V séparée** : 5 A pour des MG90S, 10 A ou plus pour des MG996R ; ajoutez ~5 A pour le poignet (bloc secteur ou BEC) |
| Main | Main imprimée en 3D à tendons, type **InMoov** (fils de pêche tirés par les servos) |
| Optionnel | Carte **PCA9685** (16 voies PWM en I2C), condensateur 1000 µF sur l'alimentation des servos |

### Câblage (Servo.h, broches par défaut)

```
                        ┌──────────────────────┐
     PC ══ USB ═════════╡  Arduino Uno / Nano  │
                        │                      │
                        │  D3  ──────────────────────── signal servo POUCE
                        │  D5  ──────────────────────── signal servo INDEX
                        │  D6  ──────────────────────── signal servo MAJEUR
                        │  D9  ──────────────────────── signal servo ANNULAIRE
                        │  D10 ──────────────────────── signal servo AURICULAIRE
                        │  D11 ──────────────────────── signal servo INCLINAISON poignet
                        │  GND ──────────┐       │
                        └────────────────│───────┘
                                         │  MASSE COMMUNE (obligatoire)
  ┌───────────────────────┐              │
  │ Alimentation 5–6 V    │ (−) ─────────┴──────────── fils MARRON/NOIRS de tous les servos
  │ séparée (≥ 5–10 A)    │ (+) ────────────────────── fils ROUGES de tous les servos
  └───────────────────────┘        └─ condensateur 1000 µF entre + et − (conseillé)
```

Sans poignet, laissez D11 libre et mettez `wrist.enabled: false`.

> ⚠️ **N'alimentez jamais les servos par la broche 5V de l'Arduino.** Leurs pics de courant
> font redémarrer la carte, voire l'endommagent. Utilisez une **alimentation séparée 5–6 V**
> et **reliez sa masse (−) au GND de l'Arduino** : sans masse commune, le signal de commande
> n'a pas de référence et les servos ne répondent pas ou tremblent.

Avec une carte **PCA9685**, câblez SDA → A4, SCL → A5, VCC → 5V Arduino (logique) et V+ →
alimentation des servos, avec les GND communs. Le code correspondant est fourni en commentaire à
la fin du sketch.

### Étapes

1. Ouvrez `firmware/hand_controller/hand_controller.ino` dans l'**IDE Arduino**. Ajustez si besoin
   `SERVO_PINS`, `OPEN_ANGLES` et `SAFE_MIN`/`SAFE_MAX`, puis téléversez-le.
2. **Fermez le moniteur série de l'IDE** : un port COM ne peut être ouvert que par un programme à la fois.
3. Repérez le port de l'Arduino (*Gestionnaire de périphériques → Ports (COM et LPT)*, ou le menu
   *Outils → Port* de l'IDE) et reportez-le dans `config.yaml` → `serial.port`.
4. Testez d'abord **sans les fils de tendons**, servos libres, en `--mode both` : comparez les
   mouvements des servos à ceux de la main 3D.
5. Réglez dans `config.yaml` :
   - `invert: true` pour un servo qui tourne à l'envers ;
   - `min_angles` / `max_angles` pour limiter la course de chaque doigt à ce que permet la mécanique
     (commencez avec une plage réduite, par exemple 20–160°) ;
   - dans le sketch, `OPEN_ANGLES` doit valoir l'angle à 0 %, c'est-à-dire `min_angles`, ou
     `max_angles` si le doigt est inversé. C'est la position de repli en cas de timeout ;
   - pour le poignet : `wrist.deviation` (`min_angle`, `max_angle`, `invert`). Dans le sketch, la
     dernière valeur de `OPEN_ANGLES` est la position neutre (milieu de la course, 90 par défaut),
     et `SAFE_MIN` / `SAFE_MAX` bornent la course mécanique.

Le firmware **ignore les trames invalides**, **limite la vitesse** des servos (`MAX_SPEED_DEG_PER_S`)
et **ouvre la main** s'il ne reçoit plus de trame pendant **1 s** (programme fermé, câble débranché…).
Côté PC, si le port est absent, `main.py` continue de fonctionner et **se reconnecte automatiquement**
dès que l'Arduino est branché.

Format des trames, en ASCII à 115200 bauds, avec des angles entiers de 0 à 180 :

| Poignet | Trame | Exemple |
|---|---|---|
| désactivé | `$pouce,index,majeur,annulaire,auriculaire\n` | `$0,90,90,180,45` |
| activé | `$pouce,index,majeur,annulaire,auriculaire,inclinaison\n` | `$0,90,90,180,45,80` |

Le firmware accepte les deux formats. Avec une trame à 5 valeurs, le poignet reste en position neutre.
Au démarrage, `main.py` affiche le nombre de valeurs envoyées.

## 8. Dépannage

**La webcam ne s'ouvre pas (« Impossible d'ouvrir la webcam n°0 »)**
- Fermez les autres applications qui l'utilisent (Teams, Zoom, Discord, onglet de navigateur…).
- Essayez une autre caméra : `python main.py --camera 1`, ou modifiez `camera.index`.
- Essayez `camera.api: msmf` dans `config.yaml`.
- Vérifiez *Paramètres Windows → Confidentialité → Caméra* : l'accès doit être autorisé pour les
  applications de bureau.

**FPS faible (< 20)**
- Un éclairage faible pousse la webcam à allonger son temps de pose, ce qui la fait descendre à
  15 FPS. C'est pourquoi `camera.exposure` vaut `-5` par défaut. Si l'image est trop sombre,
  éclairez davantage la pièce plutôt que de passer à `-4` ou `auto`.
- Restez en 640×480. Gardez `camera.api: dshow`, qui permet le MJPG à 30 FPS.
- Fermez les applications gourmandes, et branchez le PC portable sur secteur (mode performance).

**La main n'est pas détectée / les doigts sautent**
- Placez votre main entière dans l'image, paume vers la caméra, devant un fond uni et bien éclairé.
- Baissez `tracking.min_detection_confidence` (par exemple à 0.4).
- Recalibrez avec `c`.

**Un doigt ne va pas jusqu'au bout / bouge trop peu** : recalibrez en ouvrant bien la main, puis en
serrant bien le poing.

**Le poignet est décalé au repos, ou bouge alors que je ne le tourne pas**
- Recalibrez (`c`) en tenant la main exactement dans la pose que vous voulez comme neutre.
- Gardez le coude posé : un déplacement de l'avant-bras est vu comme un mouvement du poignet.
- Le poignet tremble : baissez `wrist.smoothing_alpha` (par exemple à 0.2).
- Le sens est inversé sur la vraie main : `invert: true` sur l'axe concerné.

**Port COM introuvable / « Accès refusé »**
- Le moniteur série de l'IDE Arduino (ou un autre programme) occupe le port : fermez-le.
- Le port change parfois d'un branchement à l'autre. Les ports détectés sont listés dans le terminal
  au démarrage.
- Clone Nano (puce CH340) : installez le pilote CH340.

**Les servos tremblent**
- Le plus souvent, l'alimentation est insuffisante : utilisez une alimentation séparée assez puissante,
  ajoutez un condensateur de 1000 µF et vérifiez la **masse commune**.
- Augmentez le lissage (`angles.smoothing_alpha` à 0.3) ou réduisez `MAX_SPEED_DEG_PER_S` dans le
  firmware.
- Évitez les butées : un servo qui force en fin de course grésille. Réduisez `min_angles` / `max_angles`.
- Utilisez des câbles de signal courts, ou passez à une carte PCA9685.

**PowerShell refuse `Activate.ps1`**
Lancez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utilisez directement
`.\.venv\Scripts\python.exe main.py`.

**« Port 8000 / 8765 déjà utilisé »** : une autre instance tourne encore. Fermez-la, ou changez
`sim.http_port` / `sim.ws_port`.

**La page 3D reste grise** : Three.js est chargé depuis un CDN (jsDelivr), il faut donc une connexion
Internet.

## 9. Fonctionnement et structure

### Calcul des angles (`src/finger_angles.py`)
- On utilise les **landmarks « monde »** de MediaPipe (en mètres, centrés sur la main) et uniquement
  des **angles entre vecteurs**. Le résultat ne dépend donc ni de la distance à la caméra, ni de la
  taille de la main, ni de son orientation.
- **Doigts longs** : somme des angles de pliage aux articulations MCP, PIP et DIP, mesurés entre les
  segments poignet→MCP→PIP→DIP→bout. On obtient environ 0° doigt tendu et environ 230° doigt plié,
  avec toutes les positions intermédiaires.
- **Pouce** : pliage aux articulations MCP et IP, plus un terme d'**opposition** (l'angle entre la
  direction du pouce et la base de l'auriculaire). En effet, le pouce pivote vers la paume plus qu'il
  ne plie ses phalanges.
- Flexion brute → **%** (bornes de la calibration) → **lissage EMA** → **angle servo** (min/max,
  inversion).

### Calcul du poignet (`src/wrist_angles.py`)
- Un **repère de la main** est construit avec le poignet (0), la base de l'index (5), celle du
  majeur (9) et celle de l'auriculaire (17) : axe long, direction du pouce et normale de la paume.
- **Inclinaison** : angle de l'axe long de la main dans le plan de la paume, par rapport à l'axe
  long de la pose neutre. Une rotation de la main sur elle-même ne la fausse pas (vérifié par les
  tests).

### Arborescence
```
robot-hand-sim/
├── main.py                  # boucle principale, affichage OpenCV, clavier, mode démo
├── config.yaml              # tous les réglages
├── src/
│   ├── tracker.py           # webcam + MediaPipe HandLandmarker (API Tasks), dessin du squelette
│   ├── finger_angles.py     # landmarks → flexion → % → angle servo
│   ├── wrist_angles.py      # landmarks → inclinaison du poignet
│   ├── calibration.py       # procédure « main ouverte / poing fermé » + JSON
│   ├── smoothing.py         # moyenne mobile exponentielle
│   └── backends/
│       ├── base.py          # classe Backend + HandState + MultiBackend
│       ├── sim_backend.py   # serveur HTTP + WebSocket pour la page 3D
│       └── serial_backend.py# trames série vers l'Arduino, reconnexion auto
├── web/index.html, hand.js  # visualiseur Three.js
├── firmware/hand_controller/hand_controller.ino
└── tests/                   # pytest (mains fictives générées par cinématique)
```

### Ajouter une sortie
Créez une classe qui hérite de `Backend` (`src/backends/base.py`) et implémentez `start()`,
`send(state)` et `stop()`. Ajoutez-la ensuite dans `create_backend()` (`src/backends/__init__.py`).
`send()` reçoit un `HandState` qui contient les pourcentages, les angles servo des doigts, les angles
et servos du poignet, et l'état de détection. `state.servo_frame` donne tous les angles servo dans
l'ordre de la trame.

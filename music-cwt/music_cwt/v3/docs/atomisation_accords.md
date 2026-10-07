# Atomisation des accords — comment ça marche (v3)

> Doc de référence du sous-système d'atomisation des accords. Explique le **mécanisme**
> (détection + reconstruction + garde lossless), les **choix de design**, et le **bilan
> empirique**. La spec des heads est dans `tokenization_rules_reviewed.md` (H1 `Chord_Atomic`,
> H15, H16) ; ici c'est le "comment + pourquoi".
>
> **État : socle figé en 2026-06.** Une seule extension depuis : la **relaxation H4/H5**
> (décoration commune à tout l'accord — ornement/articulation), cf. **§4bis**. Validée
> lossless (1959/1959 accords décorés round-trip exact, 24559/24559 nus sans régression).

---

## 1. Idée et objectif

Un accord de N notes occupe N positions de contexte en tokenisation séquentielle
(`Chord_Start → Chord_Cont… → Chord_End`). L'atomisation compresse un accord **reconnu**
en **un seul** compound token `H1 = Chord_Atomic`, qui porte :

- **H15** = qualité harmonique (8 valeurs : `Maj`, `min`, `dim`, `aug`, `dom7`, `dim7`, `half_dim7`, `sus4`)
- **H16** = voicing = arrangement physique (10 valeurs : inversion + spacing + doublure)
- **H3** = fondamentale (épellation kern exacte, ex. `eee-` = E♭6)
- **H2** = durée commune ; **H9** = hampes communes
- **H4 / H5** = ornement / articulation **communs à tout l'accord** (relaxation 2026-06, cf. §4bis).
  `NULL` si l'accord est nu. H6/H7/H8 restent toujours `NULL` sur un `Chord_Atomic`.

Buts visés : (a) compression de séquence ; (b) **biais inductif harmonique** (le modèle
apprend "Do majeur" comme une unité, pas 3 hauteurs indépendantes).

> **⚠️ Bilan réel (cf. §6)** : la compression est **marginale (~2 %)**. La seule valeur
> potentielle est le biais inductif, à mesurer par l'**A/B** (deux v3, avec/sans atomisation).

---

## 2. Vue d'ensemble : deux directions

```
   PARTITION (kern)                          TOKENS
   "4c 4e- 4g"  ──── DÉTECTION ────►  Chord_Atomic, H15=min, H16=close_root_nodb, H3=c
   (tokeniseur, _try_atomize_chord)

   Chord_Atomic, H15, H16, H3 ──── RECONSTRUCTION ────►  "4c 4e- 4g"
   (postprocess, _atomic_chord_to_kern → _atomic_chord_pitches)
```

**Point clé** : détection et reconstruction utilisent la **même** fonction d'épellation
(`_atomic_chord_pitches`, dans le tokeniseur, importée par postprocess). C'est ce qui
permet la garde lossless (§4) et garantit la cohérence par construction.

> **MIDI vs kern** : la **détection** fait de l'arithmétique d'intervalles sur des entiers
> de demi-tons (MIDI) — légitime, elle analyse des notes arbitraires. La **reconstruction**
> est **100 % kern** (épellation diatonique, octave par répétition de lettre) — **aucun MIDI**
> (un détour MIDI introduisait dérive d'octave + mauvaises enharmonies, supprimé 2026-06).

---

## 3. Détection (kern → H15/H16/H3)

Code : `multihead_tokenizer.py`, fonctions `_detect_chord_quality`, `_detect_chord_voicing`,
orchestrées par `_try_atomize_chord`.

### 3a. Qualité (H15) — `_detect_chord_quality(midi_pitches)`
- Calcule les classes de hauteur distinctes (pitch-classes mod 12).
- Pour **chaque** pc candidat comme fondamentale, calcule le set d'intervalles et le compare
  aux 8 motifs de `_QUALITY_INTERVALS` (ex. `Maj = {0,4,7}`, `dom7 = {0,4,7,10}`).
- **Préfère la fondamentale en basse** (position fondamentale) ; sinon premier candidat.
- Retourne `(root_midi, qname)` où `root_midi` = la plus basse note ayant la pc fondamentale.

### 3b. Voicing (H16) — `_detect_chord_voicing(midi_pitches, root_midi)`
- **Spread** = (note haute − note basse) en demi-tons : `<12` → `close`, `<24` → `open`, `≥24` → hors-liste (`None`).
- **Inversion** depuis l'intervalle (basse − fondamentale) mod 12 : `0`→`root`, `3-4`→`inv1`,
  `6-8`→`inv2`, `10-11`→`inv3`, **sinon** → hors-liste (`None`). (C'est pourquoi un `sus4`
  avec la **quarte** à la basse, intervalle 5, est rejeté : "inv1" = tierce à la basse, or sus4 n'a pas de tierce.)
- **Doublure** par comptage de pc : `nodb` (aucune doublée), `dbR` (fondamentale doublée),
  `dbX` (une note non-fondamentale doublée), sinon hors-liste.
- Construit le label `{spread}_{inversion}_{doubling}` et vérifie qu'il est dans `H16_VOCAB`.

### 3c. Fondamentale (H3)
On reprend l'**orthographe enharmonique exacte** du membre dont le MIDI = `root_midi`
(préserve D♯ vs E♭). Stockée telle quelle dans H3.

---

## 4. La garde lossless — le cœur de la correction (2026-06)

`_try_atomize_chord` n'atomise un accord **que si TOUTES** ces conditions tiennent :

1. ≥ 3 notes.
2. **H6 (tie), H7 (slur), H8 (phrase)** : aucune, sur aucun membre. **H4 (ornement) /
   H5 (articulation)** : autorisés **uniquement si COMMUNS** à tous les membres
   (sinon fallback) — cf. §4bis.
3. Hampes (H9) et beams (H10) communs à tous les membres.
4. Durée (H2) commune, non-NULL.
5. Qualité dans les 8 (H15) **et** voicing dans les 10 (H16).
6. **GARDE LOSSLESS (hauteurs)** : `_atomic_chord_pitches(H15, H16, H3)` reproduit
   **exactement** les notes écrites (orthographe + octave). Sinon → `None` → séquentiel.
7. **GARDE LOSSLESS (décoration)** : le suffixe re-émis pour (H4, H5) se re-parse
   exactement vers (H4, H5). Sinon → `None` → séquentiel. (§4bis)

**Pourquoi la condition 6 est indispensable.** H15/H16 sont des **catégories grossières** :
plusieurs accords écrits différemment matchent la même paire (H15, H16) —
- **enharmonie** : `mi#-sol#-do` vs `fa-la♭-do` (mêmes sons, écriture ≠) ;
- **registre d'un voicing ouvert** : `open` ne fixe pas l'octave des notes non-fondamentales ;
- **note doublée d'un `dbX`** : la catégorie ne dit pas *laquelle* ;
- **fondamentale d'un accord symétrique** (`aug`/`dim7`) : plusieurs racines valides.

Sans garde, la reconstruction canonique **ré-épellerait ou re-placerait** des notes →
**changement de partition**. La garde renvoie ces cas vers le séquentiel (qui préserve
l'écrit exact). Résultat **vérifié kern→token→kern sur 24.6 k accords : 0 partition altérée.**

---

## 4bis. Relaxation H4/H5 — décoration commune à l'accord (2026-06)

**Motivation.** La règle d'origine (« aucune décoration per-note ») renvoyait au séquentiel
tout accord portant un ornement/articulation, alors que le cas dominant est une décoration
**identique sur tous les membres** : un staccato/accent/fermata/arpège posé sur **tout le bloc**
(kern l'écrit sur chaque note de l'accord). On peut alors la porter sur le token atomique sans
perte.

**Règle.** Pour H4 (ornement) et H5 (articulation) **séparément** : atomisation autorisée si la
valeur est **commune à tous les membres** (toutes `NULL`, ou toutes la même valeur). La valeur
commune est stockée dans H4/H5 du `Chord_Atomic` et **ré-émise sur chaque membre** à la
reconstruction (`_atomic_chord_decoration`, partagée tokeniseur↔postprocess). Valeurs
**différentes** entre membres → fallback séquentiel.

**Pourquoi H6/H7/H8 restent exclus.**
- **H6 (tie)** est per-note hétérogène : il lie une **hauteur précise** à l'événement suivant
  (ex. note du haut tenue, basses re-frappées) → non représentable par une valeur unique.
- **H7 (slur) / H8 (phrase)** sont portés par **un seul** membre ; comme la reconstruction
  **réordonne** les notes (ascendant), on ne saurait pas re-attribuer le marqueur au bon membre.

**Losslessness.** Décoration commune ⇒ ré-émission sur chaque membre ⇒ même **multiset**
`(hauteur, ornement, articulation)`. La garde lossless (cond. 7) re-parse le suffixe pour
confirmer (H4, H5) ↦ (H4, H5). **Vérifié : 1959/1959 accords décorés et 24559/24559 nus
round-trip exact** (local Chopin + KernScores). Décorations vues : arpège `:` (64 %),
staccato `'` (26 %), accent `^`, fermata `;`, doux `` ` ``.

**Côté modèle.** H4/H5 ne sont **plus** masqués à `NULL` sur `Chord_Atomic` dans la loss
(`cwt_model.py`) ; seuls H6/H7/H8 le restent. H4/H5 sont donc appris normalement sur les
accords atomiques (Stage 3).

---

## 5. Reconstruction (H15/H16/H3 → kern) — 100 % kern

Code : `_atomic_chord_pitches(h15, h16, h3)` (tokeniseur) ; `_atomic_chord_to_kern` (postprocess)
y ajoute la durée H2, le tracker d'accidentels et les hampes H9.

### 5a. Épellation des notes (diatonique, sans MIDI)
Chaque qualité a une liste de degrés `_QUALITY_DEGREES[h15] = [(pas_diatonique, intervalle_demi-tons), …]`
(ex. `Maj = [(0,0),(2,4),(4,7)]` = fonda, tierce majeure +2 lettres/+4 demi-tons, quinte +4 lettres/+7).
Pour chaque degré :
- **lettre** = lettre_fonda décalée de `pas_diatonique` (mod 7) ;
- **accidentel** = `(pc_cible − pc_naturel_de_la_lettre) mod 12`, ramené dans `[-6, +6]`.

→ donne l'épellation correcte (ex. `dom7` sur D → `f#`, pas `g♭`).

### 5b. Placement des octaves (diatonique, JAMAIS pitch-class)
> Règle critique : l'octave se calcule en **diatonique** (indice de lettre), pas en pitch-class.
> Un C♭ a pc 11 (élevé) mais c'est une note **basse** — comparer par pc casserait le registre.

- **Position fondamentale serrée** : octave de chaque note = `octave_fonda + (indice_lettre_fonda + pas_diatonique) // 7`. La fondamentale reste à son octave H3.
- **Inversion** : on descend d'une octave les notes à partir de l'indice de basse (la basse passe sous la fondamentale, qui reste ancrée).
- **Open** : on monte d'une octave la **plus basse note non-fondamentale au-dessus de la basse** (3ce en position fonda, 5te en inv1, 3ce en inv2…). La fondamentale n'est **jamais** déplacée → reste à son octave H3.
- **Doublure** : `dbR` = ajoute la fondamentale une octave au-dessus de sa position ; `dbX` = double la **note de basse de l'inversion** (la plus souvent doublée en pratique).

Chaque note est ensuite rendue par `_render_kern_pitch(lettre, accidentel, octave)` (octave =
répétition de lettre : `cc` = C5, `CC` = C2).

### 5c. Propriété de point fixe
Ces choix canoniques sont des **points fixes de la détection** : reconstruire un `(H15,H16,H3)`
donne des notes qui re-détectent **exactement** le même `(H15,H16,H3)`. Vérifié sur les
**100** combinaisons voicing × racines (10 racines, dont enharmoniques). C'est cette propriété
qui rend la garde lossless efficace (ce qui peut s'atomiser, s'atomise de façon stable).

---

## 6. Bilan empirique (corpus local Chopin + KernScores, 2026-06)

### Taux réel d'atomisation
- **21.4 % des accords 3+ notes** s'atomisent = **1.25 % de tous les évènements** → **~2 % de compression de séquence**. (Le "−11 à −13 %" de l'étude initiale `accords_atomiques.md` était trop optimiste.)

### Plafond irréductible (pourquoi 78 % des accords 3+ ne s'atomisent pas)
| Blocage | % des accords 3+ |
|---|---|
| Décorés (ties / ornements / slurs sur un membre) | 32 % |
| **Qualité hors-liste** (dont **98 % de clusters non-tertiens** : suspensions, agrégats, quartal) | 22 % |
| Hampes / beams mixtes entre membres | 20 % |
| ATOMISÉS | **21.4 %** |
| lossless-fail (enharmonie / symétrie / registre) | 2 % |
| voicing hors-liste / durées mixtes | ~2 % |

### Ce qui n'améliorerait (presque) rien
- **Plus de qualités** (Maj7, min7, 9es…) : 98 % des accords hors-liste sont **non-tertiens** → +0.5 % seulement.
- **Plus de voicings** : +1 % (les échecs restants sont enharmonie/symétrie, non adressables par le voicing).
- → **21 % est le plafond pratique.** L'atomisation ne peut pas être étendue significativement.

### Conclusion de design
La valeur ne peut **pas** venir de la compression (~2 %). Elle ne peut venir que du **biais
inductif harmonique** → à trancher par l'**A/B** (deux v3 : atomisée vs flag "tout séquentiel").
Cf. `../../../todo.md` § A/B accords atomiques.

---

## 7. Historique des bugs corrigés (2026-06)

Trouvés via le test round-trip répété (idempotence) ; **aucun** n'altérait la partition en
production grâce à la garde lossless, mais ils faisaient tomber des accords valides en séquentiel :

1. **Détour MIDI → dérive d'octave + mauvaise enharmonie** (`g♭` au lieu de `f#`). → reconstruction réécrite 100 % kern.
2. **`dbX` doublait toujours la 3ce** ; en inversion on double la **basse** (5te en inv2). → open_inv2_dbX : 0 % → 90 %.
3. **`open` montait la mauvaise note** (la 5te/sommet au lieu de la 3ce). → règle unifiée "monter la plus basse non-fonda" ; open_root : 0 % → 85 %, open_inv1 : 0 % → 95 %.

Total : taux d'atomisation des accords reconnus passé de ~80 % → **91 %**, toujours **0 partition altérée**.

---

## 8. Où vit le code

| Fichier | Rôle |
|---|---|
| `v3/tokenizers/multihead_tokenizer.py` | `_QUALITY_INTERVALS`, `_detect_chord_quality`, `_detect_chord_voicing`, `_try_atomize_chord` (+ garde lossless hauteurs + décoration), `_QUALITY_DEGREES`, `_parse_root_kern`, `_render_kern_pitch`, **`_atomic_chord_pitches`** (reconstruction partagée), **`_atomic_chord_decoration`** (suffixe H4/H5 partagé, §4bis) |
| `v3/postprocess.py` | `_atomic_chord_to_kern` (appelle `_atomic_chord_pitches` + `_atomic_chord_decoration` + durée + tracker accidentels + hampes) |
| `v3/cwt_model.py` | masque de loss : H15/H16 actifs uniquement si `H1 = Chord_Atomic` ; H6/H7/H8 forcés `NULL` sur `Chord_Atomic` (H4/H5 **non** masqués depuis §4bis) |
| `v3/docs/tokenization_rules_reviewed.md` | spec des heads H1/H15/H16, règle de fallback |

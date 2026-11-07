# Data

## What is (and is not) in this repository

| | Included? | Why |
|---|---|---|
| Twitter/X dataset (users, posts, follower graphs) | **No** | Post text cannot be redistributed under the X developer policy. The dataset is cited in the paper. |
| 2024+ posts of the 50 users of the unseen-data evaluation | **No** | Collected with the official X API v2 user-timeline endpoint; same policy. |
| Task I sampled datasets (`dataset_*.json`) and per-post predictions | **No** | They contain post text. Aggregate metrics for all 15 settings are in `../results/task1/`. |
| `vapor_user_annotations.json` | **Yes** | Our manual annotations of the 383 VAPOR users (user ID + occupation L1–L4 + interests/hobbies). No posts, names or bios. |
| `vapor_annotation_summary.xlsx` | Yes | Label counts / summary of the annotations. |
| `iab_interest_label_mapping.json` | Yes | id ↔ label mapping of the IAB interest categories used by the fine-tuned classifiers. |
| User-study material | **No** | Only aggregate response counts (`../results/user_study/`). |

## Expected layout of the local dataset

The scripts read three files/folders, configured through environment variables (`../.env.example`):

```
$VAPOR_DESCRIPTIONS  user_descriptions_rounds.xlsx   one sheet per weekly round; columns include
                                                     user_id, description, followers_count, following, verified
$VAPOR_TWEETS        tweets_no_duplicates.csv        user_id, text, tweet_created_at ("%a %b %d %H:%M:%S %z %Y"),
                                                     retweeted_status_text, Round
$VAPOR_GRAPHS/       gr_week_<round>.pkl             pickled networkx DiGraph of the follower network of that week
                                                     (edge u -> v means u follows v; node ids are strings)
```

With these in place every script in `task1_*`, `task2_*` and `task3_*` runs end-to-end. The annotation file used by Task III (`selected_users2_fully_labeled.json`) is `vapor_user_annotations.json` joined with each user's original posts and bio from the dataset (see `task2_post_generation/user_selection_with_tweets.py`).

## Taxonomies

* Occupations: [2018 Standard Occupational Classification](https://www.bls.gov/soc/2018/) (U.S. Bureau of Labor Statistics) – L1 = major group, L2 = minor group, L3 = broad occupation, L4 = detailed occupation.
* Interests/hobbies: [IAB Tech Lab Content Taxonomy v3.1](https://iabtechlab.com/standards/content-taxonomy/) – tier-1 categories.

# Clustering threshold — measured results

Produced by `python -m pipeline.eval_score` from `eval/labeled_pairs.csv`.
This is the number Phase 2 rests on: how similar two articles must be before the
pipeline calls them the same event.

## What was measured

- 60 labelled pairs; article text retrieved for **99 of 99** articles.
- All three variants compared on the **60 pairs every variant could score** (an article whose page would not load has no lead and no body, so it drops out of two of the three — comparing on different subsets would flatter whichever got the easier one).

## The three variants, head to head

### headline

- Pairs scored: **60** (15 same, 43 different, 2 unsure)
- **Ranking quality (AUC): 0.980** — the chance it scores a true pair above a false one. 1.000 is perfect, 0.500 is a coin flip.
- **Overlap zone: 0.835 – 0.935** — inside it, human verdicts go both ways, so the pipeline cannot decide alone. 23% of pairs land here and would go to the Gemini Flash verifier.
- Best single threshold **0.869** — precision 93%, recall 87%, F1 0.897.
- The human's `unsure` pairs scored #3 at 0.892, #19 at 0.890: inside the overlap zone — the maths finds these hard too, which is what we hoped.

Closest calls it got wrong or nearly did:

- `different` #8 scored **0.935** — “სოფლის მეურნეობის სამინისტროში ახალი უწყება – „გასტრონომიის ” / “გარემოს დაცვისა და სოფლის მეურნეობის სამინისტროს სისტემაში ა”
- `different` #60 scored **0.856** — “კანადაში ხანძრის 190 კერას ებრძვიან” / “შოტლანდიაში ტყის ხანძრებს ებრძვიან”
- `different` #12 scored **0.842** — “ფსიქოტროპული ნივთიერების დიდი ოდენობით შემოტანის ბრალდებით ყ” / “დააკავეს კაცი, რომელმაც უცხო ქვეყნის მოქალაქე ქალს დაარტყა”
- `same` #59 scored only **0.835** — “ალექსანდრე კარტოზიას მიერ გერმანიის პრეზიდენტისთვის რწმუნება” / “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ”
- `same` #56 scored only **0.841** — “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ” / “გერმანიის საგარეო უწყების ოფიციალური ვებ გვერდიდან საქართველ”
- `same` #11 scored only **0.869** — “ქორწილის ინციდენტში ბრალდებული გირაოს სანაცვლოდ გათავისუფლდა” / “გიორგი ჭიღლაძე 5 000-ლარიანი გირაოს სანაცვლოდ სასამართლო დარ”

### headline+lead

- Pairs scored: **60** (15 same, 43 different, 2 unsure)
- **Ranking quality (AUC): 0.989** — the chance it scores a true pair above a false one. 1.000 is perfect, 0.500 is a coin flip.
- **Overlap zone: 0.828 – 0.921** — inside it, human verdicts go both ways, so the pipeline cannot decide alone. 15% of pairs land here and would go to the Gemini Flash verifier.
- Best single threshold **0.828** — precision 83%, recall 100%, F1 0.909.
- The human's `unsure` pairs scored #3 at 0.884, #19 at 0.869: inside the overlap zone — the maths finds these hard too, which is what we hoped.

Closest calls it got wrong or nearly did:

- `different` #8 scored **0.921** — “სოფლის მეურნეობის სამინისტროში ახალი უწყება – „გასტრონომიის ” / “გარემოს დაცვისა და სოფლის მეურნეობის სამინისტროს სისტემაში ა”
- `different` #33 scored **0.850** — “ლევან მახაშვილი: გერმანია-საქართველოს ურთიერთობა სხვა რეჟიმშ” / “ლევან მახაშვილი: ტრამპის წერილის შემდეგ შეიძლება ოპოზიციას ძ”
- `different` #7 scored **0.828** — “დიდი ოდენობით თანხის თაღლითურად დაუფლების ბრალდებით თბილისში” / “19 ტონა ვადაგასული ხორცპროდუქტის რეალიზაციის ბრალდებით 1 პირ”
- `same` #59 scored only **0.828** — “ალექსანდრე კარტოზიას მიერ გერმანიის პრეზიდენტისთვის რწმუნება” / “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ”
- `same` #56 scored only **0.848** — “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ” / “გერმანიის საგარეო უწყების ოფიციალური ვებ გვერდიდან საქართველ”
- `same` #13 scored only **0.889** — “დააკავეს კაცი, რომელმაც კახეთში ქალს სცემა” / “დააკავეს კაცი, რომელმაც უცხო ქვეყნის მოქალაქე ქალს დაარტყა”

### headline+body

- Pairs scored: **60** (15 same, 43 different, 2 unsure)
- **Ranking quality (AUC): 0.997** — the chance it scores a true pair above a false one. 1.000 is perfect, 0.500 is a coin flip.
- **Overlap zone: 0.880 – 0.887** — inside it, human verdicts go both ways, so the pipeline cannot decide alone. 5% of pairs land here and would go to the Gemini Flash verifier.
- Best single threshold **0.880** — precision 94%, recall 100%, F1 0.968.
- The human's `unsure` pairs scored #3 at 0.906, #19 at 0.920: **outside the overlap zone** — worth a look: the maths is confident where a human wasn't.

Closest calls it got wrong or nearly did:

- `different` #8 scored **0.887** — “სოფლის მეურნეობის სამინისტროში ახალი უწყება – „გასტრონომიის ” / “გარემოს დაცვისა და სოფლის მეურნეობის სამინისტროს სისტემაში ა”
- `different` #1 scored **0.843** — “რას ამბობს ქორწილის ინციდენტში მონაწილე ქალი რუსეთიდან” / “თელავის სასტუმროში მომხდარ ინციდენტში მონაწილე უცხო ქვეყნის ”
- `different` #58 scored **0.839** — “თელავის სასტუმროში მომხდარ ინციდენტში მონაწილე უცხო ქვეყნის ” / “"არ ვთვლი, რომ რაიმე პროვოკაცია მოვაწყვეთ"  — "აგარანი მამულ”
- `same` #59 scored only **0.880** — “ალექსანდრე კარტოზიას მიერ გერმანიის პრეზიდენტისთვის რწმუნება” / “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ”
- `same` #56 scored only **0.886** — “მაკა ბოჭორიშვილი: პოზიტიური მომენტია, როდესაც ჩვენს ელჩს ექნ” / “გერმანიის საგარეო უწყების ოფიციალური ვებ გვერდიდან საქართველ”
- `same` #16 scored only **0.889** — “ენდი ბერნემი ოფიციალურად გახდა გაერთიანებული სამეფოს პრემიერ” / “ენდი ბერნემი გაერთიანებული სამეფოს პრემიერ-მინისტრი გახდა”

## Read this before trusting the numbers

- **Only 15 pairs are `same`.** Every percentage here rests on those 15, so treat the
  thresholds as roughly right, not precise. The fix is a second labelling round with
  candidates chosen *by these embeddings* — far better at surfacing real matches than
  the deliberately dumb word-overlap that picked round 1.
- **This set is not a natural sample of Georgian news.** It was built dense with hard
  cases on purpose, so a good score here does not translate to the same score in
  production. It is a tuning instrument, not a report card.
- **Pairs are not clusters.** A threshold that judges two articles well can still chain
  in production: A matches B, B matches C, and A and C end up in one event without ever
  having matched. That is a real risk to handle when clustering is wired in, not here.

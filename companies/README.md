# Company Profiles

Company profiles give `interviewkit prep` the values, principles, and
interview themes it should use when generating questions.

## File Format

Create one markdown file per company:

```text
companies/<company>.md
```

Use this structure:

```markdown
# Company: Example

## Interview Focus

Short notes about the company's behavioral interview style.

## Principles Or Competencies

### Principle Name

What this principle means and what a strong story should show.

## Notes

Additional public interview-prep notes.
```

The CLI resolves `--company amazon` to `companies/amazon.md`, so keep filenames
lowercase and hyphenated.

## Included Profiles

- `amazon.md`: Amazon's 16 Leadership Principles.
- `microsoft.md`: six common Microsoft behavioral competencies, plus culture
  signals for growth mindset, inclusion, and customer impact.

## Public Sources

- Amazon Leadership Principles:
  https://www.aboutamazon.com/working-at-amazon/our-leadership-principles
- Microsoft interview tips:
  https://careers.microsoft.com/v2/global/en/hiring-tips/interview-tips.html
- Microsoft culture:
  https://careers.microsoft.com/us/en/culture

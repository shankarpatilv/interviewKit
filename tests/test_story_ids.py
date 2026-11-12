from interviewkit.ingest.story_ids import slugify, story_id


def test_slugify_strips_numbering_and_punctuation() -> None:
    assert (
        slugify("5. SNS → SQS → Lambda Over Step Functions") == "sns-sqs-lambda-over-step-functions"
    )
    assert slugify("Nova Premier Over Nova Lite/Pro") == "nova-premier-over-nova-lite-pro"


def test_story_id_uses_available_headers() -> None:
    assert story_id("amazon.md", "Technical Decisions", "6. DynamoDB Over RDS") == (
        "amazon/technical-decisions/dynamodb-over-rds"
    )
    assert story_id("amazon.md", "Role") == "amazon/role"
    assert story_id("dir/amazon.md") == "dir/amazon/overview"
    assert story_id("amazon.md", None, "Context") == "amazon/context"

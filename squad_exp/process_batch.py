import argparse
import json
import os

from openai import OpenAI


def get_client(api_key_file):
    if api_key_file:
        os.environ["OPENAI_API_KEY"] = open(api_key_file).read().strip()
    return OpenAI()


def print_status(batch):
    print(f"batch_id={batch.id}")
    print(f"status={batch.status}")
    print(f"request_counts={batch.request_counts}")
    if batch.error_file_id:
        print(f"error_file_id={batch.error_file_id}")
    if batch.output_file_id:
        print(f"output_file_id={batch.output_file_id}")


def retrieve_batch(client, batch_id):
    batch = client.batches.retrieve(batch_id)
    print_status(batch)
    return batch


def extract_text(row):
    if row.get("error"):
        return None
    response = row.get("response") or {}
    if response.get("status_code") not in (None, 200):
        return None
    body = response.get("body") or {}
    if body.get("output_text"):
        return body["output_text"].strip()
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text") and content.get("text"):
                return content["text"].strip()
    return None


def load_results(client, batch):
    content = client.files.content(batch.output_file_id)
    results = {}
    failed = 0
    for line in content.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = extract_text(row)
        if not text:
            failed += 1
            continue
        results[row["custom_id"]] = text
    print(f"batch output: successful={len(results)}, failed={failed}")
    return results


def save_error_file(client, batch, out_path=None):
    if batch.errors and batch.errors.data:
        print("batch-level errors:")
        for err in batch.errors.data:
            print(f"  line={getattr(err, 'line', None)} code={err.code} message={err.message}")
    if not batch.error_file_id:
        print("No error file.")
        return
    out_path = out_path or f"{batch.id}_errors.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(client.files.content(batch.error_file_id).text)
    print(f"Wrote error file to {out_path}")


def inject_results(orig_file, results_by_id):
    with open(orig_file) as f:
        examples = json.load(f)

    injected = skipped = 0
    unmatched = set(results_by_id)
    for ex in examples:
        ex_id = ex["id"]
        text = results_by_id.get(ex_id)
        if not text:
            continue
        unmatched.discard(ex_id)
        if ex.get("generated_paraphrase", "").strip():
            skipped += 1
            continue
        ex["generated_paraphrase"] = text
        injected += 1

    if injected:
        with open(orig_file, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
    print(
        f"{orig_file}: injected={injected}, skipped_existing={skipped}, "
        f"unmatched_ids={len(unmatched)}"
    )
    if unmatched:
        print(f"unmatched sample ids: {', '.join(list(unmatched)[:5])}")


def submit_batch(client, filename):
    with open(filename, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"{filename}: batch_id={batch.id}, status={batch.status}")


def main(args):
    client = get_client(args.api_key_file)

    if args.command == "submit":
        submit_batch(client, args.jsonl)
        return

    if args.command == "cancel":
        print_status(client.batches.cancel(args.batch_id))
        return

    if args.command == "status":
        batch = retrieve_batch(client, args.batch_id)
        save_error_file(client, batch, args.error_out)
        return

    if args.command == "download":
        if not args.orig_file:
            raise SystemExit("download requires --orig-file")
        batch = retrieve_batch(client, args.batch_id)
        if batch.status != "completed":
            print("Batch not completed; skip download.")
            return
        if not batch.output_file_id:
            print("No output file.")
            return
        inject_results(args.orig_file, load_results(client, batch))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_file", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    submit_parser = sub.add_parser("submit", help="Submit a batch JSONL file")
    submit_parser.add_argument("jsonl", metavar="JSONL")

    status_parser = sub.add_parser("status", help="Check batch status and fetch error file if present")
    status_parser.add_argument("batch_id", metavar="batch_id")
    status_parser.add_argument("--error-out", default="errors.jsonl", help="Where to save error JSONL")

    cancel_parser = sub.add_parser("cancel", help="Cancel an in-progress batch")
    cancel_parser.add_argument("batch_id", metavar="batch_id")

    download_parser = sub.add_parser("download", help="Download completed output and inject into original file")
    download_parser.add_argument("batch_id", metavar="batch_id")
    download_parser.add_argument("--orig-file", help="train.json or valid.json to inject into")

    main(parser.parse_args())

# usage: python process_batch.py status 1234567890 --error-out errors.jsonl
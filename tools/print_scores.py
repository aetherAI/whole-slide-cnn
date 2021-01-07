import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--column",
        type=str,
        help="One of filename, is_adeno, is_squamous, adeno or squamous.",
    )
    parser.add_argument(
        "test_result",
        type=str,
    )
    args = parser.parse_args()

    with open(args.test_result) as f:
        content = json.load(f)

    content = sorted(content, key=lambda x: x["slide_path"])

    data = []
    for row in content:
        if args.column == "filename":
            data.append(row["slide_path"].split("/")[-1])
        elif args.column == "is_adeno":
            data.append(1 if row["y_true"] == 1 else 0)
        elif args.column == "is_squamous":
            data.append(1 if row["y_true"] == 2 else 0)
        elif args.column == "adeno":
            data.append(row["y_pred"][1])
        elif args.column == "squamous":
            data.append(row["y_pred"][2])
        else:
            raise NotImplementedError()

    for datum in data:
        print(datum)


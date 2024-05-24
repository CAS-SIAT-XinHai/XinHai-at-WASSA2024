import argparse
import json
import logging

from wassa.evaluators import EVALUATOR_REGISTRY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WASSA evaluator")
    tasks = ['wassa2023', 'wassa2024']
    parser.add_argument("-t", "--task", default="wassa2023", type=str,
                        help="one of: {}".format(", ".join(sorted(tasks))))
    parser.add_argument("-td", "--task_dir", default="evaluations/llmeval", type=str,
                        help="data directory")
    parser.add_argument("-pd", "--prompts_dir", default="prompts", type=str,
                        help="prompts directory")
    splits = ['dev', 'test']
    parser.add_argument("-s", "--split", default="validation", type=str,
                        help="one of: {}".format(", ".join(sorted(splits))))
    methods = ['baseline']
    parser.add_argument("-m", "--method", default="data", type=str,
                        help="Methods used for generation.")

    parser.add_argument('--n_shot', type=int, default=0, help='bar help')

    parser.add_argument('--model_name', type=str, help='bar help')
    parser.add_argument('--model_api_key', type=str, help='bar help')
    parser.add_argument('--model_api_base', type=str, help='bar help')

    parser.add_argument('--evaluator_name', type=str, help='bar help')
    parser.add_argument('--evaluator_api_key', type=str, help='bar help')
    parser.add_argument('--evaluator_api_base', type=str, help='bar help')

    parser.add_argument('--debug', store_ture=True, help='bar help')

    args = parser.parse_args()

    print(args)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    evaluator = EVALUATOR_REGISTRY[args.task][args.method](
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_api_base=args.model_api_base,
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_api_base=args.evaluator_api_base,
            task_dir=args.task_dir,
        )

    evaluator.run(args.split, args.n_shot, num_retries=5)

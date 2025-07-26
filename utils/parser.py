import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SRVSKG")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="movie", help="Choose a dataset:[movie,amazon-book,yelp]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--reg_weight', type=float, default=1e-2, help='Weight_regularization.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--topK', type=int, default=20, help='Workers number.')
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--ua_node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20,50]', help='Output sizes of every layer')




    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_virtual", type=int, default=3, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--n_iter', type=int, default=3, help='number of n_iter')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    # parser.add_argument('--data_path', default='amazon-book', help='Dataset path')
    parser.add_argument('--save_file', default='test1231', help='Filename')

    # DICE
    parser.add_argument('--neg_sample_rate', type=int, default=3, help='Negative Sampling Ratio.')
    parser.add_argument('--margin_for_neg', type=int, default=40, help='Margin for negative sampling.')
    parser.add_argument('--pool', type=int, default=40, help='Pool for negative sampling.')
    parser.add_argument('--dis_pen', type=float, default=0.01, help='Discrepency penalty.')
    parser.add_argument('--int_weight', type=float, default=0.01, help='Weight for interest term.')
    parser.add_argument('--pop_weight', type=float, default=0.01, help='Weight for popularity term.')

    parser.add_argument('--model',type=str,default='my_model',help='type for model')
    parser.add_argument('--alpha_sign', type=float, default=-0.2)
    parser.add_argument('--offset', type=float, default=4.0)
    parser.add_argument('--eigs_dim', type=int, default=64)
    parser.add_argument('--lambda_reg', type=float, default=1e-4)
    parser.add_argument('--n_layers_sig', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--beta_sign', type=float, default=1.0, help='Control negtice impact.')

    return parser.parse_args()

args = parse_args()

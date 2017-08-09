from collections import Iterable


def parse_lambda_expr_helper(s, offset):
    if s[offset] != '(':
        raise ValueError('malformed string: node did not start with open paren at position %d' % offset)

    offset += 2
    # extract node name
    name = ''
    while s[offset] != ' ':
        name += s[offset]
        offset += 1
    node = Node(name)

    # extract child nodes
    while True:
        if s[offset] != ' ':
            raise ValueError('malformed string: node should have either had a '
                             'close paren or a space at position %d' % offset)

        offset += 1
        if s[offset] == ')':
            offset += 1
            return node, offset
        elif s[offset] == '(':
            child_node, offset = parse_lambda_expr_helper(s, offset)
        else:
            child_name = ''
            while offset < len(s) and s[offset] != ' ':
                child_name += s[offset]
                offset += 1

            child_node = Node(child_name)

        node.add_child(child_node)


def parse_lambda_expr(s):
    return parse_lambda_expr_helper(s, 0)[0]


class Node(object):
    def __init__(self, name, children=None):
        self.name = name
        self.parent = None
        self.children = list()
        if children:
            if isinstance(children, Iterable):
                for child in children:
                    self.add_child(child)
            elif isinstance(children, Node):
                self.add_child(children)
            else: raise ValueError('Wrong type for child nodes')

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def __hash__(self):
        return NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if len(self.children) != len(other.children):
            return False

        if self.name == 'and' or self.name == 'or':
            return sorted(self.children, key=lambda x: x.name) == sorted(other.children, key=lambda x:x.name)
        else:
            return self.children == other.children

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Node[%s, %d children]' % (self.name, len(self.children))

if __name__ == '__main__':
    # lambda_expr_str = 'lambda $0 e ( exists $1 ( and ( oneway $1 ) ( from $1 ci0 ) ( to $1 ci1 ) ( day_number $1 dn0 ) ( month $1 mn0 ) ( = ( fare $1 ) $0 ) ) )'
    # lambda_expr_str = 'ap0 daf asdf asdf'
    # lambda_expr = parse_lambda_expr(lambda_expr_str, 0)
    # file_path = '/Users/yinpengcheng/Research/lang2logic/seq2tree/atis/data/train.txt'
    # for line in open(file_path):
    #     entry = line.strip().split('\t')
    #     lambda_expr_str = entry[1]
    #     lambda_expr = parse_lambda_expr_helper(lambda_expr_str, 0)
    #     print lambda_expr

    # lf1 = parse_lambda_expr('( fb0 )')
    # lf2 = parse_lambda_expr('( f1b0 )')

    # print lf1 == lf2

    lf = ' '.join(['(', 'lambda', '$0', 'e', '(', 'and', '(', 'flight', '$0', ')', '(', 'from', '$0', 'ci0', ')', '(', 'to', '$0', 'ci1', ')', ')', ')'])
    lf = parse_lambda_expr(lf)

    # lfs = map(lambda x: x.strip().split('\t')[1], open('/Users/yinpengcheng/Research/SemanticParsing/atis-data/seq2seq_atis/train.tree_fmt.txt'))
    # for lf in lfs:
    #     parse_lambda_expr(lf)
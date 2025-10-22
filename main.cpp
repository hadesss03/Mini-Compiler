#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cctype>
#include <stdexcept>
#include <limits>
#include <memory>

using namespace std;

enum TokenType {
    T_INT, T_RETURN, T_INCLUDE, T_MAIN, T_PRINT, T_COUT, T_CIN,
    T_IF, T_ELSE, T_WHILE, T_ENDL, T_USING, T_NAMESPACE,
    T_IDENTIFIER, T_NUMBER, T_STRING, T_HEADER_NAME,
    T_ASSIGN, T_PLUS, T_MINUS, T_MUL, T_DIV,
    T_SEMI, T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE,
    T_EQ, T_NE, T_LT, T_GT, T_DOUBLE_GT, T_LE, T_GE,
    T_HASH, T_DOUBLE_LT, T_COMMA, T_EOF, T_UNKNOWN
};

struct Token {
    TokenType type;
    string value;
};

class Lexer {
    string text;
    size_t pos;
    char current_char;

public:
    Lexer(string input) : text(input), pos(0) {
        current_char = text[pos];
    }

    void advance() {
        pos++;
        current_char = (pos < text.size()) ? text[pos] : '\0';
    }

    void skip_whitespace() {
        while (isspace(current_char)) advance();
    }

    Token number() {
        string result;
        bool negative = false;

        if (current_char == '-') {
            negative = true;
            advance();
            if (current_char == '\0' || !isdigit(current_char)) {
                throw runtime_error("Invalid number format");
            }
        }

        while (isdigit(current_char)) {
            result += current_char;
            advance();
        }

        if (negative) result = "-" + result;
        return {T_NUMBER, result};
    }

    Token identifier() {
        string result;
        while (isalnum(current_char) || current_char == '_') {
            result += current_char;
            advance();
        }
        if (result == "int") return {T_INT, result};
        if (result == "return") return {T_RETURN, result};
        if (result == "include") return {T_INCLUDE, result};
        if (result == "main") return {T_MAIN, result};
        if (result == "printf") return {T_PRINT, result};
        if (result == "cout") return {T_COUT, result};
        if (result == "if") return {T_IF, result};
        if (result == "else") return {T_ELSE, result};
        if (result == "while") return {T_WHILE, result};
        if (result == "endl") return {T_ENDL, result};
        if (result == "using") return {T_USING, result};
        if (result == "namespace") return {T_NAMESPACE, result};
        if (result == "cin") return {T_CIN, result};
        return {T_IDENTIFIER, result};
    }

    Token string_literal() {
        string result;
        advance();
        while (current_char != '"' && current_char != '\0') {
            if (current_char == '\\') {
                advance();
                switch (current_char) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case '\\': result += '\\'; break;
                    case '"': result += '"'; break;
                    default: result += '\\'; result += current_char; break;
                }
                advance();
            } else {
                result += current_char;
                advance();
            }
        }
        advance();
        return {T_STRING, result};
    }

    vector<Token> tokenize() {
        vector<Token> tokens;
        while (current_char != '\0') {
            if (isspace(current_char)) {
                skip_whitespace();
            } else if (isdigit(current_char) || (current_char == '-' && isdigit(text[pos+1]))) {
                tokens.push_back(number());
            } else if (isalpha(current_char) || current_char == '_') {
                tokens.push_back(identifier());
            } else if (current_char == '"') {
                tokens.push_back(string_literal());
            } else if (current_char == '#') {
                tokens.push_back({T_HASH, "#"});
                advance();
                string directive;
                while (isalpha(current_char)) {
                    directive += current_char;
                    advance();
                }
                if (directive == "include") {
                    tokens.push_back({T_INCLUDE, "include"});
                    skip_whitespace();
                    if (current_char == '<' || current_char == '"') {
                        char delim = current_char;
                        advance();
                        string header;
                        while (current_char != delim && current_char != '\0' && current_char != '\n') {
                            header += current_char;
                            advance();
                        }
                        if (current_char == delim) advance();
                        tokens.push_back({T_HEADER_NAME, header});
                    }
                    while (current_char != '\n' && current_char != '\0') advance();
                }
            } else {
                switch (current_char) {
                    case '=':
                        advance();
                        if (current_char == '=') {
                            tokens.push_back({T_EQ, "=="});
                            advance();
                        } else {
                            tokens.push_back({T_ASSIGN, "="});
                        }
                        break;
                    case '!':
                        advance();
                        if (current_char == '=') {
                            tokens.push_back({T_NE, "!="});
                            advance();
                        } else {
                            tokens.push_back({T_UNKNOWN, "!"});
                        }
                        break;
                    case '<':
                        advance();
                        if (current_char == '<') {
                            tokens.push_back({T_DOUBLE_LT, "<<"});
                            advance();
                        } else if (current_char == '=') {
                            tokens.push_back({T_LE, "<="});
                            advance();
                        } else {
                            tokens.push_back({T_LT, "<"});
                        }
                        break;
                    case '>':
                        advance();
                        if (current_char == '>') {
                            tokens.push_back({T_DOUBLE_GT, ">>"});
                            advance();
                        } else if (current_char == '=') {
                            tokens.push_back({T_GE, ">="});
                            advance();
                        } else {
                            tokens.push_back({T_GT, ">"});
                        }
                        break;
                    case '+': tokens.push_back({T_PLUS, "+"}); advance(); break;
                    case '-': tokens.push_back({T_MINUS, "-"}); advance(); break;
                    case '*': tokens.push_back({T_MUL, "*"}); advance(); break;
                    case '/': tokens.push_back({T_DIV, "/"}); advance(); break;
                    case ';': tokens.push_back({T_SEMI, ";"}); advance(); break;
                    case '(': tokens.push_back({T_LPAREN, "("}); advance(); break;
                    case ')': tokens.push_back({T_RPAREN, ")"}); advance(); break;
                    case '{': tokens.push_back({T_LBRACE, "{"}); advance(); break;
                    case '}': tokens.push_back({T_RBRACE, "}"}); advance(); break;
                    case ',': tokens.push_back({T_COMMA, ","}); advance(); break;
                    default:
                        tokens.push_back({T_UNKNOWN, string(1, current_char)});
                        advance();
                        break;
                }
            }
        }
        tokens.push_back({T_EOF, ""});
        return tokens;
    }
};

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(class ASTVisitor& visitor) const = 0;
};

class ExpressionNode : public ASTNode {};
class StatementNode : public ASTNode {};

class ProgramNode : public ASTNode {
public:
    vector<unique_ptr<StatementNode>> statements;
    void accept(ASTVisitor& visitor) const override;
};

class NumberNode : public ExpressionNode {
public:
    int value;
    NumberNode(int val) : value(val) {}
    void accept(ASTVisitor& visitor) const override;
};

class StringNode : public ExpressionNode {
public:
    string value;
    StringNode(const string& val) : value(val) {}
    void accept(ASTVisitor& visitor) const override;
};

class IdentifierNode : public ExpressionNode {
public:
    string name;
    IdentifierNode(const string& id) : name(id) {}
    void accept(ASTVisitor& visitor) const override;
};

class BinaryOpNode : public ExpressionNode {
public:
    TokenType op;
    unique_ptr<ExpressionNode> left;
    unique_ptr<ExpressionNode> right;
    BinaryOpNode(TokenType op_, unique_ptr<ExpressionNode> l, unique_ptr<ExpressionNode> r)
        : op(op_), left(move(l)), right(move(r)) {}
    void accept(ASTVisitor& visitor) const override;
};

class UnaryOpNode : public ExpressionNode {
public:
    TokenType op;
    unique_ptr<ExpressionNode> operand;
    UnaryOpNode(TokenType op_, unique_ptr<ExpressionNode> o) : op(op_), operand(move(o)) {}
    void accept(ASTVisitor& visitor) const override;
};

class AssignmentNode : public StatementNode {
public:
    string identifier;
    unique_ptr<ExpressionNode> expr;
    AssignmentNode(const string& id, unique_ptr<ExpressionNode> e)
        : identifier(id), expr(move(e)) {}
    void accept(ASTVisitor& visitor) const override;
};

class DeclarationNode : public StatementNode {
public:
    string identifier;
    unique_ptr<ExpressionNode> initExpr;
    DeclarationNode(const string& id, unique_ptr<ExpressionNode> expr)
        : identifier(id), initExpr(move(expr)) {}
    void accept(ASTVisitor& visitor) const override;
};

class BlockNode : public StatementNode {
public:
    vector<unique_ptr<StatementNode>> statements;
    BlockNode(vector<unique_ptr<StatementNode>> stmts) : statements(move(stmts)) {}
    void accept(ASTVisitor& visitor) const override;
};

class IfStatementNode : public StatementNode {
public:
    unique_ptr<ExpressionNode> condition;
    unique_ptr<BlockNode> thenBlock;
    unique_ptr<BlockNode> elseBlock;
    IfStatementNode(unique_ptr<ExpressionNode> cond, unique_ptr<BlockNode> tBlock, unique_ptr<BlockNode> eBlock)
        : condition(move(cond)), thenBlock(move(tBlock)), elseBlock(move(eBlock)) {}
    void accept(ASTVisitor& visitor) const override;
};

class WhileStatementNode : public StatementNode {
public:
    unique_ptr<ExpressionNode> condition;
    unique_ptr<BlockNode> body;
    WhileStatementNode(unique_ptr<ExpressionNode> cond, unique_ptr<BlockNode> b)
        : condition(move(cond)), body(move(b)) {}
    void accept(ASTVisitor& visitor) const override;
};

class PrintNode : public StatementNode {
public:
    vector<unique_ptr<ExpressionNode>> args;
    bool isCout;
    PrintNode(vector<unique_ptr<ExpressionNode>> a, bool cout) : args(move(a)), isCout(cout) {}
    void accept(ASTVisitor& visitor) const override;
};

class InputNode : public StatementNode {
public:
    vector<string> variables;
    InputNode(vector<string> vars) : variables(move(vars)) {}
    void accept(ASTVisitor& visitor) const override;
};

class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    virtual void visit(const ProgramNode& node) = 0;
    virtual void visit(const NumberNode& node) = 0;
    virtual void visit(const StringNode& node) = 0;
    virtual void visit(const IdentifierNode& node) = 0;
    virtual void visit(const BinaryOpNode& node) = 0;
    virtual void visit(const UnaryOpNode& node) = 0;
    virtual void visit(const AssignmentNode& node) = 0;
    virtual void visit(const DeclarationNode& node) = 0;
    virtual void visit(const BlockNode& node) = 0;
    virtual void visit(const IfStatementNode& node) = 0;
    virtual void visit(const WhileStatementNode& node) = 0;
    virtual void visit(const PrintNode& node) = 0;
    virtual void visit(const InputNode& node) = 0;
};

void ProgramNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void NumberNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void StringNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void IdentifierNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void BinaryOpNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void UnaryOpNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void AssignmentNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void DeclarationNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void BlockNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void IfStatementNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void WhileStatementNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void PrintNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
void InputNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }

class InterpreterVisitor : public ASTVisitor {
    map<string, int> variables;
    int result;
    string string_result;

public:
    void visit(const ProgramNode& node) override {
        for (const auto& stmt : node.statements) {
            stmt->accept(*this);
        }
    }

    void visit(const NumberNode& node) override {
        result = node.value;
    }

    void visit(const StringNode& node) override {
        string_result = node.value;
    }

    void visit(const IdentifierNode& node) override {
        if (!variables.count(node.name)) {
            throw runtime_error("Undefined variable: " + node.name);
        }
        result = variables[node.name];
    }

    void visit(const BinaryOpNode& node) override {
        node.left->accept(*this);
        int left = result;
        node.right->accept(*this);
        int right = result;

        switch (node.op) {
            case T_PLUS: result = left + right; break;
            case T_MINUS: result = left - right; break;
            case T_MUL: result = left * right; break;
            case T_DIV:
                if (right == 0) throw runtime_error("Division by zero");
                result = left / right;
                break;
            case T_EQ: result = left == right; break;
            case T_NE: result = left != right; break;
            case T_LT: result = left < right; break;
            case T_GT: result = left > right; break;
            case T_LE: result = left <= right; break;
            case T_GE: result = left >= right; break;
            default: throw runtime_error("Unsupported binary operator");
        }
    }

    void visit(const UnaryOpNode& node) override {
        node.operand->accept(*this);
        if (node.op == T_MINUS) {
            result = -result;
        } else {
            throw runtime_error("Unsupported unary operator");
        }
    }

    void visit(const AssignmentNode& node) override {
        node.expr->accept(*this);
        variables[node.identifier] = result;
    }

    void visit(const DeclarationNode& node) override {
        int value = 0;
        if (node.initExpr) {
            node.initExpr->accept(*this);
            value = result;
        }
        variables[node.identifier] = value;
    }

    void visit(const BlockNode& node) override {
        for (const auto& stmt : node.statements) {
            stmt->accept(*this);
        }
    }

    void visit(const IfStatementNode& node) override {
        node.condition->accept(*this);
        if (result) {
            node.thenBlock->accept(*this);
        } else if (node.elseBlock) {
            node.elseBlock->accept(*this);
        }
    }

    void visit(const WhileStatementNode& node) override {
        while (true) {
            node.condition->accept(*this);
            if (!result) break;
            node.body->accept(*this);
        }
    }

    void visit(const PrintNode& node) override {
        if (node.isCout) {
            for (const auto& arg : node.args) {
                if (dynamic_cast<const IdentifierNode*>(arg.get()) &&
                    static_cast<const IdentifierNode*>(arg.get())->name == "endl") {
                    cout << endl;
                } else {
                    if (dynamic_cast<const StringNode*>(arg.get())) {
                        arg->accept(*this);
                        cout << string_result;
                    } else {
                        arg->accept(*this);
                        cout << result;
                    }
                }
            }
        } else {
            if (node.args.empty()) {
                throw runtime_error("printf requires at least one argument");
            }
            node.args[0]->accept(*this);
            if (string_result.empty()) {
                throw runtime_error("printf requires a format string as the first argument");
            }
            string format = string_result;
            vector<int> args;
            for (size_t i = 1; i < node.args.size(); ++i) {
                node.args[i]->accept(*this);
                args.push_back(result);
            }
            size_t pos = 0;
            for (int arg : args) {
                pos = format.find("%d", pos);
                if (pos != string::npos) {
                    format.replace(pos, 2, to_string(arg));
                    pos += to_string(arg).size();
                }
            }
            cout << format;
        }
    }

    void visit(const InputNode& node) override {
        for (const auto& var : node.variables) {
            cout << "Enter value for " << var << ": ";
            int value;
            cin >> value;
            variables[var] = value;
        }
    }
};

class Parser {
    vector<Token> tokens;
    size_t pos;
    Token current_token;

    Token peek_next() {
        if (pos + 1 < tokens.size()) return tokens[pos + 1];
        return {T_EOF, ""};
    }

    static string tokenTypeToString(TokenType type) {
        switch (type) {
            case T_INT: return "'int'";
            case T_RETURN: return "'return'";
            case T_INCLUDE: return "'include'";
            case T_MAIN: return "'main'";
            case T_PRINT: return "'printf'";
            case T_COUT: return "'cout'";
            case T_IF: return "'if'";
            case T_ELSE: return "'else'";
            case T_WHILE: return "'while'";
            case T_ENDL: return "'endl'";
            case T_USING: return "'using'";
            case T_NAMESPACE: return "'namespace'";
            case T_CIN: return "'cin'";
            case T_IDENTIFIER: return "identifier";
            case T_NUMBER: return "number";
            case T_STRING: return "string";
            case T_HEADER_NAME: return "header name";
            case T_ASSIGN: return "'='";
            case T_PLUS: return "'+'";
            case T_MINUS: return "'-'";
            case T_MUL: return "'*'";
            case T_DIV: return "'/'";
            case T_SEMI: return "';'";
            case T_LPAREN: return "'('";
            case T_RPAREN: return "')'";
            case T_LBRACE: return "'{'";
            case T_RBRACE: return "'}'";
            case T_EQ: return "'=='";
            case T_NE: return "'!='";
            case T_LT: return "'<'";
            case T_GT: return "'>'";
            case T_DOUBLE_GT: return "'>>'";
            case T_LE: return "'<='";
            case T_GE: return "'>='";
            case T_HASH: return "'#'";
            case T_DOUBLE_LT: return "'<<'";
            case T_COMMA: return "','";
            case T_EOF: return "end of file";
            default: return "unknown token";
        }
    }

public:
    Parser(vector<Token> toks) : tokens(toks), pos(0) {
        current_token = tokens[pos];
    }

    void advance() {
        pos++;
        if (pos < tokens.size()) current_token = tokens[pos];
    }

    void eat(TokenType type) {
        if (current_token.type == type) {
            advance();
        } else {
            string expected = tokenTypeToString(type);
            string found = current_token.value.empty()
                ? tokenTypeToString(current_token.type)
                : "'" + current_token.value + "'";
            throw runtime_error("Expected " + expected + " but found " + found);
        }
    }

    unique_ptr<ExpressionNode> parse_expression() {
        return parse_comparison();
    }

    unique_ptr<ExpressionNode> parse_comparison() {
        auto left = parse_add_sub();
        while (current_token.type >= T_EQ && current_token.type <= T_GE) {
            TokenType op = current_token.type;
            eat(op);
            auto right = parse_add_sub();
            left = make_unique<BinaryOpNode>(op, move(left), move(right));
        }
        return left;
    }

    unique_ptr<ExpressionNode> parse_add_sub() {
        auto left = parse_term();
        while (current_token.type == T_PLUS || current_token.type == T_MINUS) {
            TokenType op = current_token.type;
            eat(op);
            auto right = parse_term();
            left = make_unique<BinaryOpNode>(op, move(left), move(right));
        }
        return left;
    }

    unique_ptr<ExpressionNode> parse_term() {
        auto left = parse_factor();
        while (current_token.type == T_MUL || current_token.type == T_DIV) {
            TokenType op = current_token.type;
            eat(op);
            auto right = parse_factor();
            left = make_unique<BinaryOpNode>(op, move(left), move(right));
        }
        return left;
    }

    unique_ptr<ExpressionNode> parse_factor() {
        if (current_token.type == T_NUMBER) {
            int val = stoi(current_token.value);
            eat(T_NUMBER);
            return make_unique<NumberNode>(val);
        }
        else if (current_token.type == T_STRING) {
            string val = current_token.value;
            eat(T_STRING);
            return make_unique<StringNode>(val);
        }
        else if (current_token.type == T_IDENTIFIER) {
            string name = current_token.value;
            eat(T_IDENTIFIER);
            return make_unique<IdentifierNode>(name);
        }
        else if (current_token.type == T_LPAREN) {
            eat(T_LPAREN);
            auto expr = parse_expression();
            eat(T_RPAREN);
            return expr;
        }
        else if (current_token.type == T_MINUS) {
            eat(T_MINUS);
            auto expr = parse_factor();
            return make_unique<UnaryOpNode>(T_MINUS, move(expr));
        }
        throw runtime_error("Unexpected token in factor: " + current_token.value);
    }

    unique_ptr<BlockNode> parse_block() {
        eat(T_LBRACE);
        vector<unique_ptr<StatementNode>> stmts;
        while (current_token.type != T_RBRACE && current_token.type != T_EOF) {
            stmts.push_back(parse_statement());
        }
        eat(T_RBRACE);
        return make_unique<BlockNode>(move(stmts));
    }

    unique_ptr<StatementNode> parse_statement() {
        if (current_token.type == T_INT) {
            eat(T_INT);
            vector<unique_ptr<DeclarationNode>> decls;
            do {
                string var = current_token.value;
                eat(T_IDENTIFIER);
                unique_ptr<ExpressionNode> init = nullptr;
                if (current_token.type == T_ASSIGN) {
                    eat(T_ASSIGN);
                    init = parse_expression();
                }
                decls.push_back(make_unique<DeclarationNode>(var, move(init)));
                if (current_token.type != T_COMMA) break;
                eat(T_COMMA);
            } while (true);
            eat(T_SEMI);
            auto declStmt = make_unique<BlockNode>(vector<unique_ptr<StatementNode>>());
            for (auto& decl : decls) {
                declStmt->statements.push_back(move(decl));
            }
            return declStmt;
        } else if (current_token.type == T_IDENTIFIER) {
            string var = current_token.value;
            eat(T_IDENTIFIER);
            eat(T_ASSIGN);
            auto expr = parse_expression();
            eat(T_SEMI);
            return make_unique<AssignmentNode>(var, move(expr));
        } else if (current_token.type == T_IF) {
            eat(T_IF);
            eat(T_LPAREN);
            auto cond = parse_expression();
            eat(T_RPAREN);
            auto thenBlock = parse_block();
            unique_ptr<BlockNode> elseBlock = nullptr;
            if (current_token.type == T_ELSE) {
                eat(T_ELSE);
                elseBlock = parse_block();
            }
            return make_unique<IfStatementNode>(move(cond), move(thenBlock), move(elseBlock));
        } else if (current_token.type == T_WHILE) {
            eat(T_WHILE);
            eat(T_LPAREN);
            auto cond = parse_expression();
            eat(T_RPAREN);
            auto body = parse_block();
            return make_unique<WhileStatementNode>(move(cond), move(body));
        } else if (current_token.type == T_PRINT || current_token.type == T_COUT) {
            bool isCout = current_token.type == T_COUT;
            eat(isCout ? T_COUT : T_PRINT);
            vector<unique_ptr<ExpressionNode>> args;
            if (current_token.type == T_LPAREN) {
                eat(T_LPAREN);
                if (current_token.type != T_RPAREN) {
                    args.push_back(parse_expression());
                    while (current_token.type == T_COMMA) {
                        eat(T_COMMA);
                        args.push_back(parse_expression());
                    }
                }
                eat(T_RPAREN);
            } else if (isCout) {
                while (current_token.type == T_DOUBLE_LT) {
                    eat(T_DOUBLE_LT);
                    if (current_token.type == T_STRING) {
                        args.push_back(make_unique<StringNode>(current_token.value));
                        eat(T_STRING);
                    } else if (current_token.type == T_ENDL) {
                        args.push_back(make_unique<IdentifierNode>("endl"));
                        eat(T_ENDL);
                    } else {
                        args.push_back(parse_expression());
                    }
                }
            }
            eat(T_SEMI);
            return make_unique<PrintNode>(move(args), isCout);
        } else if (current_token.type == T_CIN) {
            eat(T_CIN);
            vector<string> vars;
            while (current_token.type == T_DOUBLE_GT) {
                eat(T_DOUBLE_GT);
                if (current_token.type == T_IDENTIFIER) {
                    vars.push_back(current_token.value);
                    eat(T_IDENTIFIER);
                }
            }
            eat(T_SEMI);
            return make_unique<InputNode>(move(vars));
        } else if (current_token.type == T_RETURN) {
            eat(T_RETURN);
            auto expr = parse_expression();
            eat(T_SEMI);
            return make_unique<BlockNode>(vector<unique_ptr<StatementNode>>());
        }
        throw runtime_error("Unknown statement: " + current_token.value);
    }

    unique_ptr<ProgramNode> parse_program() {
        auto program = make_unique<ProgramNode>();
        while (current_token.type != T_EOF) {
            if (current_token.type == T_USING) {
                eat(T_USING);
                eat(T_NAMESPACE);
                eat(T_IDENTIFIER);
                eat(T_SEMI);
            } else if (current_token.type == T_HASH) {
                eat(T_HASH);
                if (current_token.type == T_INCLUDE) {
                    eat(T_INCLUDE);
                    if (current_token.type == T_HEADER_NAME) {
                        eat(T_HEADER_NAME);
                    }
                }
            } else if (current_token.type == T_INT && peek_next().type == T_MAIN) {
                eat(T_INT);
                eat(T_MAIN);
                eat(T_LPAREN);
                eat(T_RPAREN);
                auto mainBlock = parse_block();
                for (auto& stmt : mainBlock->statements) {
                    program->statements.push_back(move(stmt));
                }
            } else {
                program->statements.push_back(parse_statement());
            }
        }
        return program;
    }
};

int main() {
    int choice;
    do {
        cout << "\n========== C++ Compiler ==========\n";
        cout << "1. Compile new file\n";
        cout << "2. Show Credits\n";
        cout << "3. Exit\n";
        cout << "====================================\n";
        cout << "Enter your choice: ";

        if (!(cin >> choice)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a number.\n";
            continue;
        }

        switch(choice) {
            case 1: {
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "\nEnter your C/C++ code (type 'END' on a new line to finish):\n";
                stringstream buffer;
                string line;
                while (getline(cin, line)) {
                    if (line == "END") break;
                    buffer << line << '\n';
                }
                cin.clear();
                string input = buffer.str();
                if (input.empty()) {
                    cerr << "Error: No input provided.\n";
                    break;
                }

                try {
                    Lexer lexer(input);
                    vector<Token> tokens = lexer.tokenize();
                    Parser parser(tokens);
                    unique_ptr<ProgramNode> ast = parser.parse_program();
                    InterpreterVisitor interpreter;
                    ast->accept(interpreter);
                } catch (exception& e) {
                    cerr << "Error: " << e.what() << endl;
                }
                break;
            }
            case 2:
                cout << "\n======= Credits =======\n";
                cout << "Hi my name is Josh J. Pantalunan\n";
                cout << "From BSCS 2B\n";
                cout << "This Mini Compiler have the features of:\n";
                cout << "\t - Lexical Analysis\n";
                cout << "\t - Abstract Syntax Tree\n";
                cout << "\t - Parsing \n";
                cout << "\t - Interpretation \n";
                cout << "=======================\n";
                break;
            case 3:
                cout << "Exiting program...\n";
                break;
            default:
                cout << "Invalid choice! Please select 1-3.\n";
                break;
        }
    } while (choice != 3);

    return 0;
}

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.regex.Pattern;

public class MiniCompilerGUI extends JFrame {
    private JTextArea codeArea;
    private JTextArea tokenArea;
    private JTextArea outputArea;
    private JButton compileButton;

    public MiniCompilerGUI() {
        setTitle("Mini C++ Compiler");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(900, 600);

        codeArea = new JTextArea();
        tokenArea = new JTextArea();
        tokenArea.setEditable(false);
        outputArea = new JTextArea();
        outputArea.setEditable(false);

        compileButton = new JButton("Compile and Run");

        JPanel leftPanel = new JPanel(new BorderLayout());
        leftPanel.add(new JScrollPane(codeArea), BorderLayout.CENTER);
        leftPanel.add(compileButton, BorderLayout.SOUTH);

        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.add(new JLabel("Tokens (Lexical Analysis)"), BorderLayout.NORTH);
        rightPanel.add(new JScrollPane(tokenArea), BorderLayout.CENTER);

        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(new JLabel("Output"), BorderLayout.NORTH);
        bottomPanel.add(new JScrollPane(outputArea), BorderLayout.CENTER);

        JSplitPane rightSplit = new JSplitPane(JSplitPane.VERTICAL_SPLIT, rightPanel, bottomPanel);
        rightSplit.setResizeWeight(0.5);

        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, leftPanel, rightSplit);
        mainSplit.setResizeWeight(0.5);

        add(mainSplit, BorderLayout.CENTER);

        JMenuBar menuBar = new JMenuBar();
        JMenu helpMenu = new JMenu("Help");
        JMenuItem creditsItem = new JMenuItem("Credits");
        helpMenu.add(creditsItem);
        menuBar.add(helpMenu);
        setJMenuBar(menuBar);

        compileButton.addActionListener(e -> compileAndRun());
        creditsItem.addActionListener(e -> showCredits());
    }

    private void compileAndRun() {
        String code = codeArea.getText();
        outputArea.setText("");
        tokenArea.setText("");
        try {
            Lexer lexer = new Lexer(code);
            List<Token> tokens = lexer.tokenize();
            StringBuilder tokenStr = new StringBuilder();
            for (Token t : tokens) {
                tokenStr.append(t.type).append(": ").append(t.value).append("\n");
            }
            tokenArea.setText(tokenStr.toString());

            Parser parser = new Parser(tokens);
            ProgramNode ast = parser.parse_program();

            InterpreterVisitor interpreter = new InterpreterVisitor(outputArea);
            ast.accept(interpreter);
        } catch (Exception ex) {
            outputArea.setText("Error: " + ex.getMessage());
        }
    }

    private void showCredits() {
        JOptionPane.showMessageDialog(this,
                "Hi my name is Josh J. Pantalunan\n" +
                        "From BSCS 2B\n" +
                        "This Mini Compiler have the features of:\n" +
                        "\t - Lexical Analysis\n" +
                        "\t - Abstract Syntax Tree\n" +
                        "\t - Parsing \n" +
                        "\t - Interpretation ");
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MiniCompilerGUI().setVisible(true));
    }
}

enum TokenType {
    T_INT, T_RETURN, T_INCLUDE, T_MAIN, T_PRINT, T_COUT, T_CIN,
    T_IF, T_ELSE, T_WHILE, T_ENDL, T_USING, T_NAMESPACE,
    T_IDENTIFIER, T_NUMBER, T_STRING, T_HEADER_NAME,
    T_ASSIGN, T_PLUS, T_MINUS, T_MUL, T_DIV,
    T_SEMI, T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE,
    T_EQ, T_NE, T_LT, T_GT, T_DOUBLE_GT, T_LE, T_GE,
    T_HASH, T_DOUBLE_LT, T_COMMA, T_EOF, T_UNKNOWN
}

class Token {
    TokenType type;
    String value;

    Token(TokenType type, String value) {
        this.type = type;
        this.value = value;
    }
}

class Lexer {
    private String text;
    private int pos;
    private char currentChar;

    public Lexer(String input) {
        this.text = input;
        this.pos = 0;
        updateCurrentChar();
    }

    private void updateCurrentChar() {
        currentChar = (pos < text.length()) ? text.charAt(pos) : '\0';
    }

    public void advance() {
        pos++;
        updateCurrentChar();
    }

    public void skipWhitespace() {
        while (Character.isWhitespace(currentChar)) advance();
    }

    public Token getNumber() {
        StringBuilder result = new StringBuilder();
        boolean negative = false;

        if (currentChar == '-') {
            negative = true;
            advance();
            if (!Character.isDigit(currentChar)) {
                throw new RuntimeException("Invalid number format");
            }
        }

        while (Character.isDigit(currentChar)) {
            result.append(currentChar);
            advance();
        }

        String val = (negative ? "-" : "") + result.toString();
        return new Token(TokenType.T_NUMBER, val);
    }

    public Token getIdentifier() {
        StringBuilder result = new StringBuilder();
        while (Character.isLetterOrDigit(currentChar) || currentChar == '_') {
            result.append(currentChar);
            advance();
        }
        String id = result.toString();
        switch (id) {
            case "int": return new Token(TokenType.T_INT, id);
            case "return": return new Token(TokenType.T_RETURN, id);
            case "include": return new Token(TokenType.T_INCLUDE, id);
            case "main": return new Token(TokenType.T_MAIN, id);
            case "printf": return new Token(TokenType.T_PRINT, id);
            case "cout": return new Token(TokenType.T_COUT, id);
            case "if": return new Token(TokenType.T_IF, id);
            case "else": return new Token(TokenType.T_ELSE, id);
            case "while": return new Token(TokenType.T_WHILE, id);
            case "endl": return new Token(TokenType.T_ENDL, id);
            case "using": return new Token(TokenType.T_USING, id);
            case "namespace": return new Token(TokenType.T_NAMESPACE, id);
            case "cin": return new Token(TokenType.T_CIN, id);
            default: return new Token(TokenType.T_IDENTIFIER, id);
        }
    }

    public Token getStringLiteral() {
        StringBuilder result = new StringBuilder();
        advance(); // skip opening "
        while (currentChar != '"' && currentChar != '\0') {
            if (currentChar == '\\') {
                advance();
                switch (currentChar) {
                    case 'n': result.append('\n'); break;
                    case 't': result.append('\t'); break;
                    case '\\': result.append('\\'); break;
                    case '"': result.append('"'); break;
                    default: result.append('\\').append(currentChar); break;
                }
                advance();
            } else {
                result.append(currentChar);
                advance();
            }
        }
        advance(); // skip closing "
        return new Token(TokenType.T_STRING, result.toString());
    }

    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();
        while (currentChar != '\0') {
            if (Character.isWhitespace(currentChar)) {
                skipWhitespace();
            } else if (Character.isDigit(currentChar) || (currentChar == '-' && pos + 1 < text.length() && Character.isDigit(text.charAt(pos + 1)))) {
                tokens.add(getNumber());
            } else if (Character.isLetter(currentChar) || currentChar == '_') {
                tokens.add(getIdentifier());
            } else if (currentChar == '"') {
                tokens.add(getStringLiteral());
            } else if (currentChar == '#') {
                tokens.add(new Token(TokenType.T_HASH, "#"));
                advance();
                StringBuilder directive = new StringBuilder();
                while (Character.isLetter(currentChar)) {
                    directive.append(currentChar);
                    advance();
                }
                String dir = directive.toString();
                if (dir.equals("include")) {
                    tokens.add(new Token(TokenType.T_INCLUDE, "include"));
                    skipWhitespace();
                    if (currentChar == '<' || currentChar == '"') {
                        char delim = currentChar;
                        advance();
                        StringBuilder header = new StringBuilder();
                        while (currentChar != (delim == '<' ? '>' : '"') && currentChar != '\0' && currentChar != '\n') {
                            header.append(currentChar);
                            advance();
                        }
                        if (currentChar == (delim == '<' ? '>' : '"')) advance();
                        tokens.add(new Token(TokenType.T_HEADER_NAME, header.toString()));
                    }
                    while (currentChar != '\n' && currentChar != '\0') advance();
                }
            } else {
                switch (currentChar) {
                    case '=':
                        advance();
                        if (currentChar == '=') {
                            tokens.add(new Token(TokenType.T_EQ, "=="));
                            advance();
                        } else {
                            tokens.add(new Token(TokenType.T_ASSIGN, "="));
                        }
                        break;
                    case '!':
                        advance();
                        if (currentChar == '=') {
                            tokens.add(new Token(TokenType.T_NE, "!="));
                            advance();
                        } else {
                            tokens.add(new Token(TokenType.T_UNKNOWN, "!"));
                        }
                        break;
                    case '<':
                        advance();
                        if (currentChar == '<') {
                            tokens.add(new Token(TokenType.T_DOUBLE_LT, "<<"));
                            advance();
                        } else if (currentChar == '=') {
                            tokens.add(new Token(TokenType.T_LE, "<="));
                            advance();
                        } else {
                            tokens.add(new Token(TokenType.T_LT, "<"));
                        }
                        break;
                    case '>':
                        advance();
                        if (currentChar == '>') {
                            tokens.add(new Token(TokenType.T_DOUBLE_GT, ">>"));
                            advance();
                        } else if (currentChar == '=') {
                            tokens.add(new Token(TokenType.T_GE, ">="));
                            advance();
                        } else {
                            tokens.add(new Token(TokenType.T_GT, ">"));
                        }
                        break;
                    case '+': tokens.add(new Token(TokenType.T_PLUS, "+")); advance(); break;
                    case '-': tokens.add(new Token(TokenType.T_MINUS, "-")); advance(); break;
                    case '*': tokens.add(new Token(TokenType.T_MUL, "*")); advance(); break;
                    case '/': tokens.add(new Token(TokenType.T_DIV, "/")); advance(); break;
                    case ';': tokens.add(new Token(TokenType.T_SEMI, ";")); advance(); break;
                    case '(': tokens.add(new Token(TokenType.T_LPAREN, "(")); advance(); break;
                    case ')': tokens.add(new Token(TokenType.T_RPAREN, ")")); advance(); break;
                    case '{': tokens.add(new Token(TokenType.T_LBRACE, "{")); advance(); break;
                    case '}': tokens.add(new Token(TokenType.T_RBRACE, "}")); advance(); break;
                    case ',': tokens.add(new Token(TokenType.T_COMMA, ",")); advance(); break;
                    default:
                        tokens.add(new Token(TokenType.T_UNKNOWN, String.valueOf(currentChar)));
                        advance();
                        break;
                }
            }
        }
        tokens.add(new Token(TokenType.T_EOF, ""));
        return tokens;
    }
}

interface ASTNode {
    void accept(ASTVisitor visitor);
}

abstract class ExpressionNode implements ASTNode {
}

abstract class StatementNode implements ASTNode {
}

class ProgramNode implements ASTNode {
    List<StatementNode> statements = new ArrayList<>();

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class NumberNode extends ExpressionNode {
    int value;

    NumberNode(int value) {
        this.value = value;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class StringNode extends ExpressionNode {
    String value;

    StringNode(String value) {
        this.value = value;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class IdentifierNode extends ExpressionNode {
    String name;

    IdentifierNode(String name) {
        this.name = name;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class BinaryOpNode extends ExpressionNode {
    TokenType op;
    ExpressionNode left;
    ExpressionNode right;

    BinaryOpNode(TokenType op, ExpressionNode left, ExpressionNode right) {
        this.op = op;
        this.left = left;
        this.right = right;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class UnaryOpNode extends ExpressionNode {
    TokenType op;
    ExpressionNode operand;

    UnaryOpNode(TokenType op, ExpressionNode operand) {
        this.op = op;
        this.operand = operand;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class AssignmentNode extends StatementNode {
    String identifier;
    ExpressionNode expr;

    AssignmentNode(String identifier, ExpressionNode expr) {
        this.identifier = identifier;
        this.expr = expr;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class DeclarationNode extends StatementNode {
    String identifier;
    ExpressionNode initExpr;

    DeclarationNode(String identifier, ExpressionNode initExpr) {
        this.identifier = identifier;
        this.initExpr = initExpr;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class BlockNode extends StatementNode {
    List<StatementNode> statements = new ArrayList<>();

    BlockNode(List<StatementNode> statements) {
        this.statements = statements;
    }

    BlockNode() {
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class IfStatementNode extends StatementNode {
    ExpressionNode condition;
    BlockNode thenBlock;
    BlockNode elseBlock;

    IfStatementNode(ExpressionNode condition, BlockNode thenBlock, BlockNode elseBlock) {
        this.condition = condition;
        this.thenBlock = thenBlock;
        this.elseBlock = elseBlock;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class WhileStatementNode extends StatementNode {
    ExpressionNode condition;
    BlockNode body;

    WhileStatementNode(ExpressionNode condition, BlockNode body) {
        this.condition = condition;
        this.body = body;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class PrintNode extends StatementNode {
    List<ExpressionNode> args = new ArrayList<>();
    boolean isCout;

    PrintNode(List<ExpressionNode> args, boolean isCout) {
        this.args = args;
        this.isCout = isCout;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

class InputNode extends StatementNode {
    List<String> variables = new ArrayList<>();

    InputNode(List<String> variables) {
        this.variables = variables;
    }

    @Override
    public void accept(ASTVisitor visitor) {
        visitor.visit(this);
    }
}

interface ASTVisitor {
    void visit(ProgramNode node);
    void visit(NumberNode node);
    void visit(StringNode node);
    void visit(IdentifierNode node);
    void visit(BinaryOpNode node);
    void visit(UnaryOpNode node);
    void visit(AssignmentNode node);
    void visit(DeclarationNode node);
    void visit(BlockNode node);
    void visit(IfStatementNode node);
    void visit(WhileStatementNode node);
    void visit(PrintNode node);
    void visit(InputNode node);
}

class InterpreterVisitor implements ASTVisitor {
    private Map<String, Integer> variables = new HashMap<>();
    private int result;
    private String stringResult;
    private JTextArea output;

    public InterpreterVisitor(JTextArea output) {
        this.output = output;
    }

    @Override
    public void visit(ProgramNode node) {
        for (StatementNode stmt : node.statements) {
            stmt.accept(this);
        }
    }

    @Override
    public void visit(NumberNode node) {
        result = node.value;
    }

    @Override
    public void visit(StringNode node) {
        stringResult = node.value;
    }

    @Override
    public void visit(IdentifierNode node) {
        if (!variables.containsKey(node.name)) {
            throw new RuntimeException("Undefined variable: " + node.name);
        }
        result = variables.get(node.name);
    }

    @Override
    public void visit(BinaryOpNode node) {
        node.left.accept(this);
        int left = result;
        node.right.accept(this);
        int right = result;

        switch (node.op) {
            case T_PLUS: result = left + right; break;
            case T_MINUS: result = left - right; break;
            case T_MUL: result = left * right; break;
            case T_DIV:
                if (right == 0) throw new RuntimeException("Division by zero");
                result = left / right;
                break;
            case T_EQ: result = (left == right) ? 1 : 0; break;
            case T_NE: result = (left != right) ? 1 : 0; break;
            case T_LT: result = (left < right) ? 1 : 0; break;
            case T_GT: result = (left > right) ? 1 : 0; break;
            case T_LE: result = (left <= right) ? 1 : 0; break;
            case T_GE: result = (left >= right) ? 1 : 0; break;
            default: throw new RuntimeException("Unsupported binary operator");
        }
    }

    @Override
    public void visit(UnaryOpNode node) {
        node.operand.accept(this);
        if (node.op == TokenType.T_MINUS) {
            result = -result;
        } else {
            throw new RuntimeException("Unsupported unary operator");
        }
    }

    @Override
    public void visit(AssignmentNode node) {
        node.expr.accept(this);
        variables.put(node.identifier, result);
    }

    @Override
    public void visit(DeclarationNode node) {
        int value = 0;
        if (node.initExpr != null) {
            node.initExpr.accept(this);
            value = result;
        }
        variables.put(node.identifier, value);
    }

    @Override
    public void visit(BlockNode node) {
        for (StatementNode stmt : node.statements) {
            stmt.accept(this);
        }
    }

    @Override
    public void visit(IfStatementNode node) {
        node.condition.accept(this);
        if (result != 0) {
            node.thenBlock.accept(this);
        } else if (node.elseBlock != null) {
            node.elseBlock.accept(this);
        }
    }

    @Override
    public void visit(WhileStatementNode node) {
        while (true) {
            node.condition.accept(this);
            if (result == 0) break;
            node.body.accept(this);
        }
    }

    @Override
    public void visit(PrintNode node) {
        if (node.isCout) {
            for (ExpressionNode arg : node.args) {
                if (arg instanceof IdentifierNode && ((IdentifierNode) arg).name.equals("endl")) {
                    output.append("\n");
                } else if (arg instanceof StringNode) {
                    arg.accept(this);
                    output.append(stringResult);
                } else {
                    arg.accept(this);
                    output.append(String.valueOf(result));
                }
            }
        } else {
            String format = "";
            List<Integer> args = new ArrayList<>();
            if (!node.args.isEmpty()) {
                node.args.get(0).accept(this);
                format = stringResult;
                for (int i = 1; i < node.args.size(); i++) {
                    node.args.get(i).accept(this);
                    args.add(result);
                }
            }
            int pos = 0;
            for (int arg : args) {
                pos = format.indexOf("%d", pos);
                if (pos != -1) {
                    String argStr = String.valueOf(arg);
                    format = format.substring(0, pos) + argStr + format.substring(pos + 2);
                    pos += argStr.length();
                }
            }
            output.append(format);
        }
    }

    @Override
    public void visit(InputNode node) {
        for (String var : node.variables) {
            String input = JOptionPane.showInputDialog("Enter value for " + var + ":");
            if (input == null) throw new RuntimeException("Input cancelled");
            int value = Integer.parseInt(input);
            variables.put(var, value);
        }
    }
}

class Parser {
    private List<Token> tokens;
    private int pos;
    private Token currentToken;

    public Parser(List<Token> tokens) {
        this.tokens = tokens;
        this.pos = 0;
        this.currentToken = (tokens.isEmpty()) ? null : tokens.get(0);
    }

    public void advance() {
        pos++;
        currentToken = (pos < tokens.size()) ? tokens.get(pos) : null;
    }

    public void eat(TokenType type) {
        if (currentToken != null && currentToken.type == type) {
            advance();
        } else {
            String expected = tokenTypeToString(type);
            String found = (currentToken == null || currentToken.value.isEmpty())
                    ? "end of file"
                    : "'" + currentToken.value + "'";
            throw new RuntimeException("Expected " + expected + " but found " + found);
        }
    }

    private static String tokenTypeToString(TokenType type) {
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

    private Token peekNext() {
        if (pos + 1 < tokens.size()) return tokens.get(pos + 1);
        return new Token(TokenType.T_EOF, "");
    }

    public ExpressionNode parse_expression() {
        return parse_comparison();
    }

    private ExpressionNode parse_comparison() {
        ExpressionNode left = parse_add_sub();
        while (currentToken != null && currentToken.type.ordinal() >= TokenType.T_EQ.ordinal() && currentToken.type.ordinal() <= TokenType.T_GE.ordinal()) {
            TokenType op = currentToken.type;
            eat(op);
            ExpressionNode right = parse_add_sub();
            left = new BinaryOpNode(op, left, right);
        }
        return left;
    }

    private ExpressionNode parse_add_sub() {
        ExpressionNode left = parse_term();
        while (currentToken != null && (currentToken.type == TokenType.T_PLUS || currentToken.type == TokenType.T_MINUS)) {
            TokenType op = currentToken.type;
            eat(op);
            ExpressionNode right = parse_term();
            left = new BinaryOpNode(op, left, right);
        }
        return left;
    }

    private ExpressionNode parse_term() {
        ExpressionNode left = parse_factor();
        while (currentToken != null && (currentToken.type == TokenType.T_MUL || currentToken.type == TokenType.T_DIV)) {
            TokenType op = currentToken.type;
            eat(op);
            ExpressionNode right = parse_factor();
            left = new BinaryOpNode(op, left, right);
        }
        return left;
    }

    private ExpressionNode parse_factor() {
        if (currentToken == null) throw new RuntimeException("Unexpected end of input");
        if (currentToken.type == TokenType.T_NUMBER) {
            int val = Integer.parseInt(currentToken.value);
            eat(TokenType.T_NUMBER);
            return new NumberNode(val);
        } else if (currentToken.type == TokenType.T_STRING) {
            String val = currentToken.value;
            eat(TokenType.T_STRING);
            return new StringNode(val);
        } else if (currentToken.type == TokenType.T_IDENTIFIER) {
            String name = currentToken.value;
            eat(TokenType.T_IDENTIFIER);
            return new IdentifierNode(name);
        } else if (currentToken.type == TokenType.T_LPAREN) {
            eat(TokenType.T_LPAREN);
            ExpressionNode expr = parse_expression();
            eat(TokenType.T_RPAREN);
            return expr;
        } else if (currentToken.type == TokenType.T_MINUS) {
            eat(TokenType.T_MINUS);
            ExpressionNode expr = parse_factor();
            return new UnaryOpNode(TokenType.T_MINUS, expr);
        }
        throw new RuntimeException("Unexpected token in factor: " + currentToken.value);
    }

    public BlockNode parse_block() {
        eat(TokenType.T_LBRACE);
        List<StatementNode> stmts = new ArrayList<>();
        while (currentToken != null && currentToken.type != TokenType.T_RBRACE && currentToken.type != TokenType.T_EOF) {
            stmts.add(parse_statement());
        }
        eat(TokenType.T_RBRACE);
        return new BlockNode(stmts);
    }

    public StatementNode parse_statement() {
        if (currentToken == null) throw new RuntimeException("Unexpected end of input");
        if (currentToken.type == TokenType.T_INT) {
            eat(TokenType.T_INT);
            List<DeclarationNode> decls = new ArrayList<>();
            do {
                String var = currentToken.value;
                eat(TokenType.T_IDENTIFIER);
                ExpressionNode init = null;
                if (currentToken.type == TokenType.T_ASSIGN) {
                    eat(TokenType.T_ASSIGN);
                    init = parse_expression();
                }
                decls.add(new DeclarationNode(var, init));
                if (currentToken.type != TokenType.T_COMMA) break;
                eat(TokenType.T_COMMA);
            } while (true);
            eat(TokenType.T_SEMI);
            BlockNode declStmt = new BlockNode();
            for (DeclarationNode decl : decls) {
                declStmt.statements.add(decl);
            }
            return declStmt;
        } else if (currentToken.type == TokenType.T_IDENTIFIER) {
            String var = currentToken.value;
            eat(TokenType.T_IDENTIFIER);
            eat(TokenType.T_ASSIGN);
            ExpressionNode expr = parse_expression();
            eat(TokenType.T_SEMI);
            return new AssignmentNode(var, expr);
        } else if (currentToken.type == TokenType.T_IF) {
            eat(TokenType.T_IF);
            eat(TokenType.T_LPAREN);
            ExpressionNode cond = parse_expression();
            eat(TokenType.T_RPAREN);
            BlockNode thenBlock = parse_block();
            BlockNode elseBlock = null;
            if (currentToken != null && currentToken.type == TokenType.T_ELSE) {
                eat(TokenType.T_ELSE);
                elseBlock = parse_block();
            }
            return new IfStatementNode(cond, thenBlock, elseBlock);
        } else if (currentToken.type == TokenType.T_WHILE) {
            eat(TokenType.T_WHILE);
            eat(TokenType.T_LPAREN);
            ExpressionNode cond = parse_expression();
            eat(TokenType.T_RPAREN);
            BlockNode body = parse_block();
            return new WhileStatementNode(cond, body);
        } else if (currentToken.type == TokenType.T_PRINT || currentToken.type == TokenType.T_COUT) {
            boolean isCout = currentToken.type == TokenType.T_COUT;
            eat(isCout ? TokenType.T_COUT : TokenType.T_PRINT);
            List<ExpressionNode> args = new ArrayList<>();
            if (currentToken.type == TokenType.T_LPAREN) {
                eat(TokenType.T_LPAREN);
                if (currentToken.type != TokenType.T_RPAREN) {
                    args.add(parse_expression());
                    while (currentToken.type == TokenType.T_COMMA) {
                        eat(TokenType.T_COMMA);
                        args.add(parse_expression());
                    }
                }
                eat(TokenType.T_RPAREN);
            } else if (isCout) {
                while (currentToken != null && currentToken.type == TokenType.T_DOUBLE_LT) {
                    eat(TokenType.T_DOUBLE_LT);
                    if (currentToken.type == TokenType.T_STRING) {
                        args.add(new StringNode(currentToken.value));
                        eat(TokenType.T_STRING);
                    } else if (currentToken.type == TokenType.T_ENDL) {
                        args.add(new IdentifierNode("endl"));
                        eat(TokenType.T_ENDL);
                    } else {
                        args.add(parse_expression());
                    }
                }
            }
            eat(TokenType.T_SEMI);
            return new PrintNode(args, isCout);
        } else if (currentToken.type == TokenType.T_CIN) {
            eat(TokenType.T_CIN);
            List<String> vars = new ArrayList<>();
            while (currentToken != null && currentToken.type == TokenType.T_DOUBLE_GT) {
                eat(TokenType.T_DOUBLE_GT);
                if (currentToken.type == TokenType.T_IDENTIFIER) {
                    vars.add(currentToken.value);
                    eat(TokenType.T_IDENTIFIER);
                }
            }
            eat(TokenType.T_SEMI);
            return new InputNode(vars);
        } else if (currentToken.type == TokenType.T_RETURN) {
            eat(TokenType.T_RETURN);
            parse_expression();
            eat(TokenType.T_SEMI);
            return new BlockNode(new ArrayList<>()); // Ignoring return value as in original
        }
        throw new RuntimeException("Unknown statement: " + currentToken.value);
    }

    public ProgramNode parse_program() {
        ProgramNode program = new ProgramNode();
        while (currentToken != null && currentToken.type != TokenType.T_EOF) {
            if (currentToken.type == TokenType.T_USING) {
                eat(TokenType.T_USING);
                eat(TokenType.T_NAMESPACE);
                eat(TokenType.T_IDENTIFIER);
                eat(TokenType.T_SEMI);
            } else if (currentToken.type == TokenType.T_HASH) {
                eat(TokenType.T_HASH);
                if (currentToken.type == TokenType.T_INCLUDE) {
                    eat(TokenType.T_INCLUDE);
                    if (currentToken.type == TokenType.T_HEADER_NAME) {
                        eat(TokenType.T_HEADER_NAME);
                    }
                }
            } else if (currentToken.type == TokenType.T_INT && peekNext().type == TokenType.T_MAIN) {
                eat(TokenType.T_INT);
                eat(TokenType.T_MAIN);
                eat(TokenType.T_LPAREN);
                eat(TokenType.T_RPAREN);
                BlockNode mainBlock = parse_block();
                program.statements.addAll(mainBlock.statements);
            } else {
                program.statements.add(parse_statement());
            }
        }
        return program;
    }
}
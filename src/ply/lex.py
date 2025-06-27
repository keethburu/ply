# -----------------------------------------------------------------------------
# ply: lex.py
#
# Copyright (C) 2001-2022
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Latest version: https://github.com/dabeaz/ply
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

import re
import sys
import types
import copy
import os
import inspect
from typing import Optional, Callable, Any
from re import Pattern

# This tuple contains acceptable string types
StringTypes: tuple[type[str], type[bytes]] = (str, bytes)

# This regular expression is used to match valid token names
_is_identifier: Pattern[str] = re.compile(r'^[a-zA-Z0-9_]+$')

# Exception thrown when invalid token encountered and no default error
# handler is defined.
class LexError(Exception):
    def __init__(self, message: str, s: str) -> None:
        self.args = (message,)
        self.text = s

# Token class.  This class is used to represent the tokens produced.
class LexToken(object):
    type: Optional[str]
    value: Optional[str]
    lineno: Optional[int]
    lexpos: Optional[int]
    lexer: Any
    def __init__(self) -> None:
        self.type = None
        self.value = None
        self.lineno = None
        self.lexpos = None
        self.lexer = None
    def __repr__(self) -> str:
        return f'LexToken({self.type},{self.value!r},{self.lineno},{self.lexpos})'

# This object is a stand-in for a logging object created by the
# logging module.

class PlyLogger(object):
    f: Any
    def __init__(self, f: Any) -> None:
        self.f = f

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.f.write((msg % args) + '\n')

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.f.write('WARNING: ' + (msg % args) + '\n')

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.f.write('ERROR: ' + (msg % args) + '\n')

    info: Callable[..., None] = critical
    debug: Callable[..., None] = critical

# -----------------------------------------------------------------------------
#                        === Lexing Engine ===
#
# The following Lexer class implements the lexer runtime.   There are only
# a few public methods and attributes:
#
#    input()          -  Store a new string in the lexer
#    token()          -  Get the next token
#    clone()          -  Clone the lexer
#
#    lineno           -  Current line number
#    lexpos           -  Current position in the input string
# -----------------------------------------------------------------------------

class Lexer:
    lexre: Optional[list[Any]]
    lexretext: Optional[Any]
    lexstatere: dict[str, Any]
    lexstateretext: dict[str, Any]
    lexstaterenames: dict[str, Any]
    lexstate: str
    lexstatestack: list[str]
    lexstateinfo: Optional[Any]
    lexstateignore: dict[str, str]
    lexstateerrorf: dict[str, Callable[..., Any]]
    lexstateeoff: dict[str, Callable[..., Any]]
    lexreflags: int
    lexdata: Optional[str]
    lexpos: int
    lexlen: int
    lexerrorf: Optional[Callable[..., Any]]
    lexeoff: Optional[Callable[..., Any]]
    lextokens: Optional[Any]
    lexignore: str
    lexliterals: str
    lexmodule: Optional[Any]
    lineno: int
    def __init__(self) -> None:
        self.lexre = None
        self.lexretext = None
        self.lexstatere = {}
        self.lexstateretext = {}
        self.lexstaterenames = {}
        self.lexstate = 'INITIAL'
        self.lexstatestack = []
        self.lexstateinfo = None
        self.lexstateignore = {}
        self.lexstateerrorf = {}
        self.lexstateeoff = {}
        self.lexreflags = 0
        self.lexdata = None
        self.lexpos = 0
        self.lexlen = 0
        self.lexerrorf = None
        self.lexeoff = None
        self.lextokens = None
        self.lexignore = ''
        self.lexliterals = ''
        self.lexmodule = None
        self.lineno = 1
        self.lexmatch = None
    def clone(self, object: Optional[Any] = None) -> "Lexer":
        c: Lexer = copy.copy(self)
        if object:
            newtab: dict[str, list[Any]] = {}
            for key, ritem in self.lexstatere.items():
                newre: list[Any] = []
                for cre, findex in ritem:
                    newfindex: list[Any] = []
                    for f in findex:
                        if not f or not f[0]:
                            newfindex.append(f)
                            continue
                        newfindex.append((getattr(object, f[0].__name__), f[1]))
                    newre.append((cre, newfindex))
                newtab[key] = newre
            c.lexstatere = newtab
            c.lexstateerrorf = {}
            for key, ef in self.lexstateerrorf.items():
                c.lexstateerrorf[key] = getattr(object, ef.__name__)
            c.lexmodule = object
        return c
    def input(self, s: str) -> None:
        self.lexdata = s
        self.lexpos = 0
        self.lexlen = len(s)
    def begin(self, state: str) -> None:
        if state not in self.lexstatere:
            raise ValueError(f'Undefined state {state!r}')
        self.lexre = self.lexstatere[state]
        self.lexretext = self.lexstateretext[state]
        self.lexignore = self.lexstateignore.get(state, '')
        self.lexerrorf = self.lexstateerrorf.get(state, None)
        self.lexeoff = self.lexstateeoff.get(state, None)
        self.lexstate = state
    def push_state(self, state: str) -> None:
        self.lexstatestack.append(self.lexstate)
        self.begin(state)
    def pop_state(self) -> None:
        self.begin(self.lexstatestack.pop())
    def current_state(self) -> str:
        return self.lexstate
    def skip(self, n: int) -> None:
        self.lexpos += n
    def token(self) -> Optional[LexToken]:
        # Make local copies of frequently referenced attributes
        lexpos: int    = self.lexpos
        lexlen: int    = self.lexlen
        lexignore: str = self.lexignore
        lexdata: Optional[str]  = self.lexdata

        while lexpos < lexlen:
            # This code provides some short-circuit code for whitespace, tabs, and other ignored characters
            if lexdata is not None and lexdata[lexpos] in lexignore:
                lexpos += 1
                continue
            
            
            # Look for a regular expression match
            lexre: Pattern[str]
            lexindexfunc: list[Optional[tuple[Optional[Callable[..., Any]], Optional[str]]]]
            for lexre, lexindexfunc in self.lexre:
                m: Optional[re.Match[str]] = lexre.match(lexdata, lexpos)
                if not m:
                    continue

                # Create a token for return
                tok: LexToken = LexToken()
                tok.value = m.group()
                tok.lineno = self.lineno
                tok.lexpos = lexpos

                i: Optional[int] = m.lastindex

                # Type-safe unpacking with None check
                entry: Optional[tuple[Optional[Callable[..., Any]], Optional[str]]] = lexindexfunc[i]
                if entry is None:
                    continue  # If no entry, skip to next regex
                func: Optional[Callable[..., Any]]
                tok_type: Optional[str]
                func, tok_type = entry
                tok.type = tok_type

                if not func:
                    # If no token type was set, it's an ignored token
                    if tok.type:
                        self.lexpos = m.end()
                        return tok
                    else:
                        lexpos = m.end()
                        break

                lexpos = m.end()

                # If token is processed by a function, call it

                tok.lexer = self      # Set additional attributes useful in token rules
                self.lexmatch = m
                self.lexpos = lexpos
                newtok = func(tok)
                del tok.lexer
                del self.lexmatch

                # Every function must return a token, if nothing, we just move to next token
                if not newtok:
                    lexpos    = self.lexpos         # This is here in case user has updated lexpos.
                    lexignore = self.lexignore      # This is here in case there was a state change
                    break
                return newtok
            else:
                # No match, see if in literals
                if lexdata is not None and lexpos < lexlen and lexdata[lexpos] in self.lexliterals:
                    tok = LexToken()
                    tok.value = lexdata[lexpos]
                    tok.lineno = self.lineno
                    tok.type = tok.value
                    tok.lexpos = lexpos
                    self.lexpos = lexpos + 1
                    return tok

                # No match. Call t_error() if defined.
                if self.lexerrorf:
                    tok = LexToken()
                    if self.lexdata is not None and lexpos < self.lexlen:
                        tok.value = self.lexdata[lexpos:]
                    else:
                        tok.value = None
                    tok.lineno = self.lineno
                    tok.type = 'error'
                    tok.lexer = self
                    tok.lexpos = lexpos
                    self.lexpos = lexpos
                    illegal_char: Optional[str] = None
                    illegal_slice: str = ""
                    newtok: Optional[LexToken] = self.lexerrorf(tok)
                    if lexpos == self.lexpos:
                        # Error method didn't change text position at all. This is an error.
                        if lexdata is not None and lexpos < lexlen:
                            illegal_char = lexdata[lexpos]
                            illegal_slice = lexdata[lexpos:]
                        else:
                            illegal_char = None
                            illegal_slice = ""
                        raise LexError(f"Scanning error. Illegal character {illegal_char!r}", illegal_slice)
                    lexpos = self.lexpos
                    if not newtok:
                        continue
                    return newtok

                self.lexpos = lexpos
                raise LexError(f"Illegal character {illegal_char!r} at index {lexpos}", illegal_slice)

        if self.lexeoff:
            tok = LexToken()
            tok.type = 'eof'
            tok.value = ''
            tok.lineno = self.lineno
            tok.lexpos = lexpos
            tok.lexer = self
            self.lexpos = lexpos
            newtok = self.lexeoff(tok)
            return newtok

        self.lexpos = lexpos + 1
        if self.lexdata is None:
            raise RuntimeError('No input string given with input()')
        return None

    # Iterator interface
    def __iter__(self) -> "Lexer":
        return self

    def __next__(self) -> LexToken:
        t = self.token()
        if t is None:
            raise StopIteration
        return t

# -----------------------------------------------------------------------------
#                           ==== Lex Builder ===
#
# The functions and classes below are used to collect lexing information
# and build a Lexer object from it.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# _get_regex(func)
#
# Returns the regular expression assigned to a function either as a doc string
# or as a .regex attribute attached by the @TOKEN decorator.
# -----------------------------------------------------------------------------
def _get_regex(func: Any) -> Optional[str]:
    return getattr(func, 'regex', func.__doc__)

# -----------------------------------------------------------------------------
# get_caller_module_dict()
#
# This function returns a dictionary containing all of the symbols defined within
# a caller further down the call stack.  This is used to get the environment
# associated with the yacc() call if none was provided.
# -----------------------------------------------------------------------------
def get_caller_module_dict(levels: int) -> dict[str, Any]:
    f = sys._getframe(levels)
    return { **f.f_globals, **f.f_locals }

# -----------------------------------------------------------------------------
# _form_master_re()
#
# This function takes a list of all of the regex components and attempts to
# form the master regular expression.  Given limitations in the Python re
# module, it may be necessary to break the master regex into separate expressions.
# -----------------------------------------------------------------------------
def _form_master_re(relist: list[str], reflags: int, ldict: dict[str, Any], toknames: dict[str, str]) -> tuple[list[Any], list[str], list[Any]]:
    if not relist:
        return [], [], []
    regex = '|'.join(relist)
    try:
        lexre = re.compile(regex, reflags)

        # Build the index to function map for the matching engine
        lexindexfunc = [None] * (max(lexre.groupindex.values()) + 1)
        lexindexnames = lexindexfunc[:]

        for f, i in lexre.groupindex.items():
            handle = ldict.get(f, None)
            if type(handle) in (types.FunctionType, types.MethodType):
                lexindexfunc[i] = (handle, toknames[f])
                lexindexnames[i] = f
            elif handle is not None:
                lexindexnames[i] = f
                if f.find('ignore_') > 0:
                    lexindexfunc[i] = (None, None)
                else:
                    lexindexfunc[i] = (None, toknames[f])

        return [(lexre, lexindexfunc)], [regex], [lexindexnames]
    except Exception:
        m = (len(relist) // 2) + 1
        llist, lre, lnames = _form_master_re(relist[:m], reflags, ldict, toknames)
        rlist, rre, rnames = _form_master_re(relist[m:], reflags, ldict, toknames)
        return (llist+rlist), (lre+rre), (lnames+rnames)

# -----------------------------------------------------------------------------
# def _statetoken(s,names)
#
# Given a declaration name s of the form "t_" and a dictionary whose keys are
# state names, this function returns a tuple (states,tokenname) where states
# is a tuple of state names and tokenname is the name of the token.  For example,
# calling this with s = "t_foo_bar_SPAM" might return (('foo','bar'),'SPAM')
# -----------------------------------------------------------------------------
def _statetoken(s: str, names: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    parts = s.split('_')
    for i, part in enumerate(parts[1:], 1):
        if part not in names and part != 'ANY':
            break

    if i > 1:
        states = tuple(parts[1:i])
    else:
        states = ('INITIAL',)

    if 'ANY' in states:
        states = tuple(names)

    tokenname = '_'.join(parts[i:])
    return (states, tokenname)


# -----------------------------------------------------------------------------
# LexerReflect()
#
# This class represents information needed to build a lexer as extracted from a
# user's input file.
# -----------------------------------------------------------------------------
class LexerReflect(object):
    ldict: dict[str, Any]
    error_func: Optional[Any]
    tokens: list[str]
    reflags: int
    stateinfo: dict[str, str]
    modules: set[Any]
    error: bool
    log: PlyLogger
    literals: str
    states: Optional[Any]
    toknames: dict[str, str]
    funcsym: dict[str, list[Any]]
    strsym: dict[str, list[Any]]
    ignore: dict[str, str]
    errorf: dict[str, Callable[..., Any]]
    eoff: dict[str, Callable[..., Any]]
    def __init__(self, ldict: dict[str, Any], log: Optional[PlyLogger] = None, reflags: int = 0) -> None:
        self.ldict = ldict
        self.error_func = None
        self.tokens = []
        self.reflags = reflags
        self.stateinfo = {'INITIAL': 'inclusive'}
        self.modules = set()
        self.error = False
        self.log = PlyLogger(sys.stderr) if log is None else log
    def get_all(self) -> None:
        self.get_tokens()
        self.get_literals()
        self.get_states()
        self.get_rules()
    def validate_all(self) -> bool:
        self.validate_tokens()
        self.validate_literals()
        self.validate_rules()
        return self.error
    def get_tokens(self) -> None:
        tokens = self.ldict.get('tokens', None)
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return

        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return

        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return

        self.tokens = tokens

    # Validate the tokens
    def validate_tokens(self) -> None:
        terminals = {}
        for n in self.tokens:
            if not _is_identifier.match(n):
                self.log.error(f"Bad token name {n!r}")
                self.error = True
            if n in terminals:
                self.log.warning(f"Token {n!r} multiply defined")
            terminals[n] = 1

    # Get the literals specifier
    def get_literals(self) -> None:
        self.literals = self.ldict.get('literals', '')
        if not self.literals:
            self.literals = ''

    # Validate literals
    def validate_literals(self) -> None:
        try:
            for c in self.literals:
                if not isinstance(c, StringTypes) or len(c) > 1:
                    self.log.error(f'Invalid literal {c!r}. Must be a single character')
                    self.error = True

        except TypeError:
            self.log.error('Invalid literals specification. literals must be a sequence of characters')
            self.error = True

    def get_states(self) -> None:
        self.states = self.ldict.get('states', None)
        # Build statemap
        if self.states:
            if not isinstance(self.states, (tuple, list)):
                self.log.error('states must be defined as a tuple or list')
                self.error = True
            else:
                for s in self.states:
                    if not isinstance(s, tuple) or len(s) != 2:
                        self.log.error("Invalid state specifier %r. Must be a tuple (statename,'exclusive|inclusive')", s)
                        self.error = True
                        continue
                    name, statetype = s
                    if not isinstance(name, StringTypes):
                        self.log.error('State name %r must be a string', name)
                        self.error = True
                        continue
                    if not (statetype == 'inclusive' or statetype == 'exclusive'):
                        self.log.error("State type for state %r must be 'inclusive' or 'exclusive'", name)
                        self.error = True
                        continue
                    if name in self.stateinfo:
                        self.log.error("State %r already defined", name)
                        self.error = True
                        continue
                    self.stateinfo[name] = statetype

    # Get all of the symbols with a t_ prefix and sort them into various
    # categories (functions, strings, error functions, and ignore characters)

    def get_rules(self) -> None:
        tsymbols = [f for f in self.ldict if f[:2] == 't_']

        # Now build up a list of functions and a list of strings
        self.toknames = {}        # Mapping of symbols to token names
        self.funcsym  = {}        # Symbols defined as functions
        self.strsym   = {}        # Symbols defined as strings
        self.ignore   = {}        # Ignore strings by state
        self.errorf   = {}        # Error functions by state
        self.eoff     = {}        # EOF functions by state

        for s in self.stateinfo:
            self.funcsym[s] = []
            self.strsym[s] = []

        if len(tsymbols) == 0:
            self.log.error('No rules of the form t_rulename are defined')
            self.error = True
            return

        for f in tsymbols:
            t = self.ldict[f]
            states, tokname = _statetoken(f, self.stateinfo)
            self.toknames[f] = tokname

            if hasattr(t, '__call__'):
                if tokname == 'error':
                    for s in states:
                        self.errorf[s] = t
                elif tokname == 'eof':
                    for s in states:
                        self.eoff[s] = t
                elif tokname == 'ignore':
                    line = t.__code__.co_firstlineno
                    file = t.__code__.co_filename
                    self.log.error("%s:%d: Rule %r must be defined as a string", file, line, t.__name__)
                    self.error = True
                else:
                    for s in states:
                        self.funcsym[s].append((f, t))
            elif isinstance(t, StringTypes):
                if tokname == 'ignore':
                    for s in states:
                        self.ignore[s] = t
                    if '\\' in t:
                        self.log.warning("%s contains a literal backslash '\\'", f)

                elif tokname == 'error':
                    self.log.error("Rule %r must be defined as a function", f)
                    self.error = True
                else:
                    for s in states:
                        self.strsym[s].append((f, t))
            else:
                self.log.error('%s not defined as a function or string', f)
                self.error = True

        # Sort the functions by line number
        for f in self.funcsym.values():
            f.sort(key=lambda x: x[1].__code__.co_firstlineno)

        # Sort the strings by regular expression length
        for s in self.strsym.values():
            s.sort(key=lambda x: len(x[1]), reverse=True)

    # Validate all of the t_rules collected
    def validate_rules(self) -> None:
        for state in self.stateinfo:
            # Validate all rules defined by functions

            for fname, f in self.funcsym[state]:
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                tokname = self.toknames[fname]
                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule %r has too many arguments", file, line, f.__name__)
                    self.error = True
                    continue

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule %r requires an argument", file, line, f.__name__)
                    self.error = True
                    continue

                if not _get_regex(f):
                    self.log.error("%s:%d: No regular expression defined for rule %r", file, line, f.__name__)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (fname, _get_regex(f)), self.reflags)
                    if c.match(''):
                        self.log.error("%s:%d: Regular expression for rule %r matches empty string", file, line, f.__name__)
                        self.error = True
                except re.error as e:
                    self.log.error("%s:%d: Invalid regular expression for rule '%s'. %s", file, line, f.__name__, e)
                    if '#' in _get_regex(f):
                        self.log.error("%s:%d. Make sure '#' in rule %r is escaped with '\\#'", file, line, f.__name__)
                    self.error = True

            # Validate all rules defined by strings
            for name, r in self.strsym[state]:
                tokname = self.toknames[name]
                if tokname == 'error':
                    self.log.error("Rule %r must be defined as a function", name)
                    self.error = True
                    continue

                if tokname not in self.tokens and tokname.find('ignore_') < 0:
                    self.log.error("Rule %r defined for an unspecified token %s", name, tokname)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (name, r), self.reflags)
                    if (c.match('')):
                        self.log.error("Regular expression for rule %r matches empty string", name)
                        self.error = True
                except re.error as e:
                    self.log.error("Invalid regular expression for rule %r. %s", name, e)
                    if '#' in r:
                        self.log.error("Make sure '#' in rule %r is escaped with '\\#'", name)
                    self.error = True

            if not self.funcsym[state] and not self.strsym[state]:
                self.log.error("No rules defined for state %r", state)
                self.error = True

            # Validate the error function
            efunc = self.errorf.get(state, None)
            if efunc:
                f = efunc
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule %r has too many arguments", file, line, f.__name__)
                    self.error = True

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule %r requires an argument", file, line, f.__name__)
                    self.error = True

        for module in self.modules:
            self.validate_module(module)

    # -----------------------------------------------------------------------------
    # validate_module()
    #
    # This checks to see if there are duplicated t_rulename() functions or strings
    # in the parser input file.  This is done using a simple regular expression
    # match on each line in the source code of the given module.
    # -----------------------------------------------------------------------------

    def validate_module(self, module: Any) -> None:
        try:
            lines, linen = inspect.getsourcelines(module)
        except IOError:
            return

        fre = re.compile(r'\s*def\s+(t_[a-zA-Z_0-9]*)\(')
        sre = re.compile(r'\s*(t_[a-zA-Z_0-9]*)\s*=')

        counthash = {}
        linen += 1
        for line in lines:
            m = fre.match(line)
            if not m:
                m = sre.match(line)
            if m:
                name = m.group(1)
                prev = counthash.get(name)
                if not prev:
                    counthash[name] = linen
                else:
                    filename = inspect.getsourcefile(module)
                    self.log.error('%s:%d: Rule %s redefined. Previously defined on line %d', filename, linen, name, prev)
                    self.error = True
            linen += 1

# -----------------------------------------------------------------------------
# lex(module)
#
# Build all of the regular expression rules from definitions in the supplied module
# -----------------------------------------------------------------------------
def lex(*, module: Optional[Any] = None, object: Optional[Any] = None, debug: bool = False, reflags: int = int(re.VERBOSE), debuglog: Optional[Any] = None, errorlog: Optional[Any] = None) -> Lexer:

    global lexer

    ldict = None
    stateinfo  = {'INITIAL': 'inclusive'}
    lexobj = Lexer()
    global token, input

    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)

    if debug:
        if debuglog is None:
            debuglog = PlyLogger(sys.stderr)

    # Get the module dictionary used for the lexer
    if object:
        module = object

    # Get the module dictionary used for the parser
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        ldict = dict(_items)
        # If no __file__ attribute is available, try to obtain it from the __module__ instead
        if '__file__' not in ldict:
            ldict['__file__'] = sys.modules[ldict['__module__']].__file__
    else:
        ldict = get_caller_module_dict(2)

    # Collect parser information from the dictionary
    linfo = LexerReflect(ldict, log=errorlog, reflags=reflags)
    linfo.get_all()
    if linfo.validate_all():
        raise SyntaxError("Can't build lexer")

    # Dump some basic debugging information
    if debug:
        debuglog.info('lex: tokens   = %r', linfo.tokens)
        debuglog.info('lex: literals = %r', linfo.literals)
        debuglog.info('lex: states   = %r', linfo.stateinfo)

    # Build a dictionary of valid token names
    lexobj.lextokens = set()
    for n in linfo.tokens:
        lexobj.lextokens.add(n)

    # Get literals specification
    if isinstance(linfo.literals, (list, tuple)):
        lexobj.lexliterals = type(linfo.literals[0])().join(linfo.literals)
    else:
        lexobj.lexliterals = linfo.literals

    lexobj.lextokens_all = lexobj.lextokens | set(lexobj.lexliterals)

    # Get the stateinfo dictionary
    stateinfo = linfo.stateinfo

    regexs = {}
    # Build the master regular expressions
    for state in stateinfo:
        regex_list = []

        # Add rules defined by functions first
        for fname, f in linfo.funcsym[state]:
            regex_list.append('(?P<%s>%s)' % (fname, _get_regex(f)))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", fname, _get_regex(f), state)

        # Now add all of the simple rules
        for name, r in linfo.strsym[state]:
            regex_list.append('(?P<%s>%s)' % (name, r))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", name, r, state)

        regexs[state] = regex_list

    # Build the master regular expressions

    if debug:
        debuglog.info('lex: ==== MASTER REGEXS FOLLOW ====')

    for state in regexs:
        lexre, re_text, re_names = _form_master_re(regexs[state], reflags, ldict, linfo.toknames)
        lexobj.lexstatere[state] = lexre
        lexobj.lexstateretext[state] = re_text
        lexobj.lexstaterenames[state] = re_names
        if debug:
            for i, text in enumerate(re_text):
                debuglog.info("lex: state '%s' : regex[%d] = '%s'", state, i, text)

    # For inclusive states, we need to add the regular expressions from the INITIAL state
    for state, stype in stateinfo.items():
        if state != 'INITIAL' and stype == 'inclusive':
            lexobj.lexstatere[state].extend(lexobj.lexstatere['INITIAL'])
            lexobj.lexstateretext[state].extend(lexobj.lexstateretext['INITIAL'])
            lexobj.lexstaterenames[state].extend(lexobj.lexstaterenames['INITIAL'])

    lexobj.lexstateinfo = stateinfo
    lexobj.lexre = lexobj.lexstatere['INITIAL']
    lexobj.lexretext = lexobj.lexstateretext['INITIAL']
    lexobj.lexreflags = reflags

    # Set up ignore variables
    lexobj.lexstateignore = linfo.ignore
    lexobj.lexignore = lexobj.lexstateignore.get('INITIAL', '')

    # Set up error functions
    lexobj.lexstateerrorf = linfo.errorf
    lexobj.lexerrorf = linfo.errorf.get('INITIAL', None)
    if not lexobj.lexerrorf:
        errorlog.warning('No t_error rule is defined')

    # Set up eof functions
    lexobj.lexstateeoff = linfo.eoff
    lexobj.lexeoff = linfo.eoff.get('INITIAL', None)

    # Check state information for ignore and error rules
    for s, stype in stateinfo.items():
        if stype == 'exclusive':
            if s not in linfo.errorf:
                errorlog.warning("No error rule is defined for exclusive state %r", s)
            if s not in linfo.ignore and lexobj.lexignore:
                errorlog.warning("No ignore rule is defined for exclusive state %r", s)
        elif stype == 'inclusive':
            if s not in linfo.errorf:
                linfo.errorf[s] = linfo.errorf.get('INITIAL', None)
            if s not in linfo.ignore:
                linfo.ignore[s] = linfo.ignore.get('INITIAL', '')

    # Create global versions of the token() and input() functions
    token = lexobj.token
    input = lexobj.input
    lexer = lexobj

    return lexobj

# -----------------------------------------------------------------------------
# runmain()
#
# This runs the lexer as a main program
# -----------------------------------------------------------------------------

def runmain(lexer: Optional[Lexer] = None, data: Optional[str] = None) -> None:
    if not data:
        try:
            filename = sys.argv[1]
            with open(filename) as f:
                data = f.read()
        except IndexError:
            sys.stdout.write('Reading from standard input (type EOF to end):\n')
            data = sys.stdin.read()

    if lexer:
        _input = lexer.input
    else:
        _input = input
    _input(data)
    if lexer:
        _token = lexer.token
    else:
        _token = token

    while True:
        tok = _token()
        if not tok:
            break
        sys.stdout.write(f'({tok.type},{tok.value!r},{tok.lineno},{tok.lexpos})\n')

# -----------------------------------------------------------------------------
# @TOKEN(regex)
#
# This decorator function can be used to set the regex expression on a function
# when its docstring might need to be set in an alternative way
# -----------------------------------------------------------------------------

def TOKEN(r: Any) -> Callable[[Any], Any]:
    def set_regex(f: Any) -> Any:
        if hasattr(r, '__call__'):
            f.regex = _get_regex(r)
        else:
            f.regex = r
        return f
    return set_regex

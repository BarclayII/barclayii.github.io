Title: Building your own graph library with C as a Python exntension - Introduction (Step 1)
Tags: python, c/c++, cython, swig

Recently, I was collaborating with other people working on a library dealing with graph-based data.  We started with [`networkx`](https://networkx.github.io/), which has a plethora of graph algorithms and generators.  It is good for initial development, but soon we realized that even basic operations (say, asking for predecessors of a list of nodes) are slow due to the entire data structure being handled in pure Python.  We then replaced a part of the internals with [`igraph`](http://igraph.org/python/).  While it is significantly faster than `networkx`.  But still we wish to have a more focused version which only focuses on basic graph operations, and potentially, extend the storage to cross-devices (like multiple CPU/GPUs and multiple machines), as well as having other low-level, performance-critical features.

In summary, if we need

* Performance-critical (otherwise why bother?)
* Frequent (otherwise maybe the overhead of having it in Python is bearable)
* Low-level (meaning that the package users wouldn't care a lot about the details)
* Reasonably-clear implementations (e.g. for a neural network module, even if it is performance-critical, frequent, and low-level, probably it's still better, or only possible, to write it in PyTorch/Tensorflow)

Then it's better to wrap the code into C and make a Python package.

Figuring out how to interface Python with C took me a lot more effort than I expected, so I put what I found inside this tutorial series.  Note that I may change my views in this series, so expect changes in the articles.

In this tutorial series, I will go through the implementation of a miniature graph library, which is a subset of `igraph`.  I hope to have an appropriate design that decouples the graph storage, attributes storage, and the user interfaces.  I tend to write everything in pure C, or maybe a very restricted subset of C++ (C++ has way too many features, and moreover I'm not an expert on everything).

The platform I am developing on will be POSIX-compliant (e.g. Linux or [Cygwin](http://cygwin.com/)).

### What tool to use?

[Cython Wiki](https://github.com/cython/cython/wiki/WrappingCorCpp) has a list of C-Python interfaces, and in particular I like the following:

* [Cython](http://cython.org/)
* [SWIG](http://www.swig.org/)
* [`ctypes`](https://docs.python.org/3/library/ctypes.html)

If I still have time, I will go through other fancy ways of interfacing Python with C/C++, such as the runtime from [TVM](https://docs.tvm.ai/dev/runtime.html).  But for now, let's begin with the easier ones.

Here is a list of pros and cons I found personally for the three methods:

* Cython is a typical three-layer design of interfacing: C at the bottom, Python at the top, and Cython in between.  The syntax of Cython is very similar to Python, so heavy Python developers familiar with C/C++ should feel comfortable.
* SWIG is a simple yet more general-purposed framework that can generate wrappers for *a lot of* languages, including Perl/Ruby/Javascript/Lua.  Similar to Cython, it also adopts a three-layer design.  The only difference I found is that SWIG syntax more resembles C (well, if you know Flex/Bison, then you should get the idea), hence less integrated with Python.  If developers wish to support other scripting languages then they should consider using SWIG.
* `ctypes` is probably the easiest to catch up since it literally calls C functions with Python code.  `ctypes` has its own data types that directly maps to C data types (`struct`'s, pointers, etc.).  As a result, `ctypes` code don't usually feel... "Pythonic".  The problem of this approach is that the developers themselves may have to write their own interface between Python and C if they don't wish to scatter `ctypes` code around the Python source files.  Also, since you are directly making calls to a C library, `ctypes` does not prevent you doing weird stuff (e.g. passing an integer as a `struct` pointer argument).
 
For the reasons above, I will mostly focus on Cython, and occasionally SWIG, as the tool for gluing C and Python together.

### The starting C code

First we will write some C code to provide toy implementation of a C structure called `struct graph`.  For now, it is only a structure consisting of two integer elements, with two functions creating and destroying the objects (which is called *constructor* and *destructor* in object-oriented programming), and two methods for incrementing those elements.  In later articles, I will extend it to actual methods to add vertices and edges into a graph.

```C
/*
 * include/graph.h
 */

#ifndef _GRAPH_H
#define _GRAPH_H

#include <stdint.h>

typedef uint64_t pgraph_size_t;
typedef int64_t pgraph_ssize_t;

struct graph {
	pgraph_size_t	n_vertices;
	pgraph_size_t	n_edges;
};

struct graph *graph_create(void);
int graph_destroy(struct graph *g);
pgraph_ssize_t graph_add_vertices(struct graph *g, pgraph_size_t n_vertices);
pgraph_ssize_t graph_add_edges(struct graph *g, pgraph_size_t n_edges);

#endif
```

```C
/*
 * src/graph.c
 */

#include <stdlib.h>

#include "graph.h"

struct graph *graph_create(void)
{
	struct graph *g;

	g = malloc(sizeof(*g));
	if (g == NULL)
		return g;

	g->n_vertices = 0;
	g->n_edges = 0;

	return g;
}

int graph_destroy(struct graph *g)
{
	free(g);

	return 0;
}

pgraph_ssize_t graph_add_vertices(struct graph *g, pgraph_size_t n_vertices)
{
	g->n_vertices += n_vertices;

	return n_vertices;
}

pgraph_ssize_t graph_add_edges(struct graph *g, pgraph_size_t n_edges)
{
	g->n_edges += n_edges;

	return n_edges;
}
```

To better manage the code base, we make two directories:

* `src/`, for keeping C source files.
* `include/`, for keeping C headers.

The directory tree structure will look like this:

```
.
├── include
│   └── graph.h
└── src
    └── graph.c
```

Now, we would like to expose the structure as a Python class.  How could we do that?

### Using Cython

#### The interface file

Let's go through using Cython first.

In general, Cython reads in an interface file, usually having extension `.pyx`, and outputs a C file which can be compiled into a shared library (or DLL on Windows).  Cython interface files usually need to

* Declare the C interfaces (structures/unions/functions/...) we wish to use.
* Declare the Python interfaces we wish to expose.

A simple Cython interface looks like the following:

```cython
# pgraph.pyx
from libc.stdint cimport uint64_t, int64_t

# First part is to declare the C interfaces we wish to use.
# You can see that the declarations resembles to the actual C header file
# ("include/graph.h") a lot.

# "cdef extern from" tells Cython that these declarations come from the
# header file you specified.
cdef extern from "../include/graph.h":
    # ctypedef is the Cython equivalent of C typedef.
    ctypedef uint64_t pgraph_size_t
    ctypedef int64_t pgraph_ssize_t

    # A Cython struct corresponds to a C struct.  The only difference is that
    # we don't have to cover all the members in the corresponding C struct;
    # we only need to write about the members we intend to expose to the Python
    # classes.
    struct graph:
        pgraph_size_t n_vertices
        pgraph_size_t n_edges

    # The function prototypes we wish to use.
    graph *graph_create()
    int graph_destroy(graph *g)
    pgraph_ssize_t graph_add_vertices(graph *g, pgraph_size_t n_vertices)
    pgraph_ssize_t graph_add_edges(graph *g, pgraph_size_t n_edges)


# The second part defines the actual Python class that uses the C code.
# Usually, it's called a "cdef" class.
#
# You write it as if you are writing Python code, barring a few notable
# differences:
# (1) A "cdef" class is always declared by "cdef class" (of course)
cdef class Graph(object):
    # (2) You need to declare the attributes you are going to use beforehand.
    # By default, these attributes are "private", meaning that external
    # Python code is not available to see them.  Of course, you can make
    # them public.
    cdef graph *_handle

    # (3) Any initializations involving C code is done in __cinit__() rather
    # than __init__().  This is the best place to, e.g., create a C graph
    # object (using malloc(3)) and store the pointer as a handle.
    def __cinit__(self):
        _handle = graph_create()
        if _handle is NULL:
            raise MemoryError()
        self._handle = _handle

    # (4) Any C-level finalization involving C code is done in __dealloc__(),
    # and there is no __del__() method.  Code destroying C objects (using
    # free(3)) should be here.
    def __dealloc__(self):
        if self._handle is not NULL:
            graph_destroy(self._handle)
            self._handle = NULL

    # (5) Optionally, you can enforce type checks on function arguments like
    # this.
    def add_vertices(self, pgraph_size_t n):
        return graph_add_vertices(self._handle, n)

    def add_edges(self, pgraph_size_t n):
        return graph_add_edges(self._handle, n)

    @property
    def n_vertices(self):
        return self._handle.n_vertices

    @property
    def n_edges(self):
        return self._handle.n_edges
```

Now let's put the file above in a separate directory called `cython`.  The tree will look like this:

```
.
├── cython
│   └── pgraph.pyx
├── include
│   └── graph.h
└── src
    └── graph.c
```

#### Building the package

Before actually going into details about the building systems such as [`setuptools`](https://setuptools.readthedocs.io/en/latest/) or [CMake](https://cmake.org/), let's go through the basics of building the Python package manually by entering shell commands, to grasp an idea of how the building process looks like.

1. First, we need to translate the Cython interface into a C source file:
    ```
    $ cython cython/pgraph.pyx
    ```
    This command will let Cython translate the files you have supplied into corresponding C files (`cython/pgraph.c` in this case).
2. Then, compile the Cython output as well as the C source code we have written.  Assuming that we have development files for Python 3 installed and we are using [GCC](https://www.gnu.org/software/gcc/), the commands will be something like:
    ```
    $ gcc -fpic -c -o cython/pgraph.o `pkg-config --cflags python3` cython/pgraph.c
    $ gcc -fpic -c -o src/graph.o -I./include src/graph.c
    ```
    The contents inside the backquotes <code>``</code> will be replaced by the shell with the output of the command inside, and `pkg-config` is a tool that generates appropriate C compiler and linker flags for the given package you wish to compile/link with (in our case, Python 3).
3. Finally, link the two object files together with the Python 3 library:
    ```
    $ gcc -shared -fpic -o pgraph`python3-config --extension-suffix` cython/pgraph.o src/graph.o `pkg-config --libs python3`
    ```
    On Linux, this will generate a shared library file named something like `pgraph.cpython-36m-x86_64-linux-gnu.so`.  The long suffix after `pgraph` is specified by the command `python3-config --extension-suffix`; Python automatically identifies shared libraries having such kind of suffix as "Python modules", and tries to import from that if you run `import pgraph` in Python.

#### Now let's test!

```
$ ipython
Python 3.6.6 (default, Jul 19 2018, 14:25:17) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import pgraph

In [2]: g = pgraph.Graph()

In [3]: g.n_vertices
Out[3]: 0

In [4]: g.add_vertices(8)
Out[4]: 8

In [5]: g.n_vertices
Out[5]: 8

In [6]: g.add_vertices('a')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-6-d716880dabd7> in <module>()
----> 1 g.add_vertices('a')

~/ML/python-pgraph/cython/pgraph.pyx in pgraph.Graph.add_vertices()
     54     # (5) Optionally, you can enforce type checks on function arguments like
     55     # this.
---> 56     def add_vertices(self, pgraph_size_t n):
     57         return graph_add_vertices(self._handle, n)
     58 

TypeError: an integer is required

In [7]: exit
```

#### Caveats

When importing a Python module from a shared library, Python looks for the C function `PyInit_<module-name>` in that library as the module entry call.  Cython generates the entry call by looking at the file name instead (e.g. when translating `foo.pyx`, Cython will generate the entry call named `PyInit_foo()`).  So when the two names does not match, Python import will fail:
```
$ mv pgraph.cpython-36m-x86_64-linux-gnu.so foo.cpython-36m-x86_64-linux-gnu.so 
$ ipython3
Python 3.6.6 (default, Jul 19 2018, 14:25:17) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import foo
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-1-7f58dd7fb72e> in <module>()
----> 1 import foo

ImportError: dynamic module does not define module export function (PyInit_foo)

In [2]: exit
```

### Using SWIG

#### The interface file

SWIG, while arguably more powerful than Cython (in the sense of supporting other languages, of course), is much more complicated.  The following is a SWIG interface file that does almost the same thing as the Cython one:

```swig
/*
 * swig/graph.i
 */

/*
 * First specify the module name - SWIG will write the Python file that has
 * the same name (pgraph.py in this case).
 */
%module pgraph

/* The content between %{...%} is added to the generated C code */
%{
#include "graph.h"
%}

/*
 * The first major part is to declare which C identifiers (variables,
 * structures, functions, etc.) we wish to expose.
 *
 * SWIG by default exposes every possible C identifier.  We can tune the
 * access controls *before* introducing the C prototypes.  The access controls
 * include whether an identifier is visible, how the identifier name is mapped
 * to another name in the target language, etc.
 */

/*
 * First, we would like to rename the C structure "graph" into Python class
 * "Graph", to avoid naming conflicts in generated Python code.  More details
 * will be explained in the "%extend" section below.
 */
%rename("Graph") "graph";
/*
 * The next thing is to hide all the C identifiers containing "graph_", since
 * they are internal functions for use in the C library.
 */
%ignore "graph_";
/*
 * We also wish to set the number of vertices and edges immutable in Python.
 */
%immutable graph::n_vertices;
%immutable graph::n_edges;
/*
 * And also, tell SWIG to use our constructor and destructor instead of its
 * own.  By default SWIG will call calloc(3) to allocate a dynamic object
 * during construction, and free(3) to deallocate it.
 */
%nodefaultctor;
%nodefaultdtor;

/*
 * Recall that we have a bunch of typedef's in "graph.h".  SWIG does not
 * follow the #include statements inside the headers as GCC does, so we
 * need to include the typedef statements for types in stdint.h by
 * ourselves.  This is done by including the builtin SWIG header "stdint.i".
 *
 * NOTE: SWIG documentation mentions that you can force SWIG to follow
 * #include statements, while making it search in the system include
 * directory, i.e. adding the following flags:
 * "-I/usr/include -includeall"
 * You SHOULD NOT do this, since SWIG often get confused when reading system
 * headers directly.
 */
%include "stdint.i"
/* Finally, steal the declaration of "struct graph" and its functions */
%include "graph.h"

/*
 * Now we need to decide how we are going to actually expose "struct graph".
 * Here, we are extending the C structure "struct graph" into a class-like
 * semantic.
 *
 * Yes, this is very similar to C++.  In fact, using C++ on SWIG will make our
 * lives much easier.  However,
 * (1) If one can make SWIG work on C, then one can literally do anything.
 * (2) C++ has a lot of features and is evolving constantly, while SWIG only
 *     supports most of C++11 (but pretty maturely).
 */
%extend graph {
	/* A constructor for C structure looks like this. */
	graph(void)
	{
		struct graph *g = graph_create();
		/*
		 * NOTE: we don't have a check for NULL pointers here.  This
		 * is dangerous, but we are not considering that for the
		 * purpose of demonstration.  More on this in later articles.
		 *
		 * Coincidentally, this is a reason why using C++ for SWIG
		 * is (likely) easier.
		 */
		return g;
	}
	/* A destructor looks like this. */
	~graph()
	{
		/*
		 * $self refers to the pointer to the current object itself
		 * (like "this" in C++ or "self" in Python).
		 */
		graph_destroy($self);
	}
	/*
	 * If we don't specify a body for a class method, SWIG will look for
	 * the C function <class-name>_<method-name>, inserting $self as the
	 * first argument.
	 *
	 * For example, the following is equivalent to
	 * pgraph_ssize_t add_vertices(pgraph_size_t n)
	 * {
	 * 	return graph_add_vertices($self, n);
	 * }
	 */
	pgraph_ssize_t add_vertices(pgraph_size_t n);
	pgraph_ssize_t add_edges(pgraph_size_t n);
};
```

#### Building the package

The building process is almost the same as Cython, except that the output of SWIG is different:

1. Translate the SWIG interface to two files: a C source file for low-level interaction between C library and Python C API, and a Python module for high-level stuff on the library built:
    ```
    $ swig -python -I./include swig/graph.i
    ```
2. Compile our C code as well as the `graph_wrap.c` generated by SWIG:
    ```
    $ gcc -fpic -c -I./include -o src/graph.o src/graph.c
    $ gcc -fpic -c -o swig/graph_wrap.o -I./include `pkg-config --cflags python3` swig/graph_wrap.c
    ```
3. Linking
    ```
    gcc -shared -fpic -o _pgraph.so src/graph.o swig/graph_wrap.o `pkg-config --libs python3`
    ```
4. Moving the generated python file `swig/pgraph.py` to the same directory as the linked shared library `_pgraph.so`.
    ```
    mv swig/pgraph.py .
    ```

#### Test!

```
ipython3
Python 3.6.6 (default, Jul 19 2018, 14:25:17) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import pgraph

In [2]: g = pgraph.Graph()

In [3]: g.n_vertices
Out[3]: 0

In [4]: g.add_vertices(8)
Out[4]: 8

In [5]: g.n_vertices
Out[5]: 8

In [6]: g.add_vertices('a')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-6-d716880dabd7> in <module>()
----> 1 g.add_vertices('a')

~/ML/python-pgraph/pgraph.py in add_vertices(self, n)
    119 
    120     def add_vertices(self, n):
--> 121         return _pgraph.Graph_add_vertices(self, n)
    122 
    123     def add_edges(self, n):

TypeError: in method 'Graph_add_vertices', argument 2 of type 'pgraph_size_t'

In [7]: exit
```

### Conclusion

We have gone through a miniature example of writing a C library as a Python extension, using both Cython and SWIG.  In the upcoming articles, I will be

* Extending the C library itself with actual functionalities on graphs.
* Updating Cython and/or SWIG interface files accordingly.
* Setting up a build environment using `setuptools` and/or CMake.
* Maybe go through either C++ or `ctypes`.

Have fun.

![import antigravity](https://www.explainxkcd.com/wiki/images/f/fd/python.png)

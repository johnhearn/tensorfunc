package example1

object FunctionalGraph {
    @JvmStatic
    fun main(args: Array<String>) {

        val inputs = constant(2.0)

        val weights = get_variable("weights", constant(3.0))
        val biases = get_variable("bias", constant(5.0))

        val graph = sigma(inputs * weights + biases)

        val y = graph()
        println(y) // forward pass

        /*
        val labels = constant(2.5)
        val loss = mean_squared_error(labels)

        println(del1(loss)(del0(graph)(inputs))) // backward pass
        */
    }
}

fun del0(f: NullaryOp): (NullaryOp) -> NullaryOp {
    return when (f) {
        ::sigma -> ::sigmaprime
        else -> throw UnsupportedOperationException("Function $f does not have derivative.")
    }
}

fun del1(f: (NullaryOp) -> NullaryOp): (NullaryOp) -> (NullaryOp) -> NullaryOp {
    return when (f) {
        ::mean_squared_error -> ::mean_squared_errorprime
        else -> throw UnsupportedOperationException("Function $f does not have derivative.")
    }
}

fun mean_squared_error(labels: NullaryOp): (NullaryOp) -> NullaryOp {
    return { predictions -> (labels - predictions) * (labels - predictions) }
}

fun mean_squared_errorprime(labels: NullaryOp): (NullaryOp) -> NullaryOp {
    return { predictions -> labels - predictions }
}

// Nullary operator
fun constant(x: Double): () -> Tenser {
    return { Tenser(x) }
}

fun get_variable(name: String, initialValue: () -> Tenser): () -> Variable {
    return { Variable(initialValue()) }
}

fun sigma(f: () -> Tenser): () -> Tenser {
    return { f().sigma() }
}

// Unary operations
fun sigmaprime(x: () -> Tenser): () -> Tenser {
    return { x().sigmaprime() }
}

// Binary operations
operator fun (() -> Tenser).plus(rhs: () -> Tenser): () -> Tenser {
    return { this() + rhs() }
}

operator fun (() -> Tenser).minus(rhs: () -> Tenser): () -> Tenser {
    return { this() - rhs() }
}

operator fun (() -> Tenser).times(rhs: () -> Tenser): () -> Tenser {
    return { this() * rhs() }
}

open class Tenser internal constructor(private val value: Double = 1.0) {

    constructor(x: Tenser) : this(x.value)

    operator fun plus(rhs: Tenser) = Tenser(this.value + rhs.value)

    operator fun times(rhs: Tenser) = Tenser(this.value * rhs.value)

    operator fun minus(rhs: Tenser) = Tenser(this.value - rhs.value)

    fun sigma() = Tenser(if (this.value < 0) 0.0 else this.value)

    fun sigmaprime(): Tenser {
        return Tenser(if (this.value < 0) 0.0 else 1.0)
    }

    override fun equals(other: Any?): Boolean {
        return other is Tenser && other.value == this.value
    }

    override fun hashCode(): Int {
        return value.hashCode()
    }

    override fun toString(): String {
        return "T($value)"
    }
}

class Variable(x: Tenser) : Tenser(x)

typealias NullaryOp = () -> Tenser
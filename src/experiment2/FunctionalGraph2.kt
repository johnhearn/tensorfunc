package experiment2

import org.junit.Test
import java.lang.reflect.Method
import kotlin.reflect.jvm.javaMethod
import kotlin.test.assertEquals

class FunctionalGraph2Should {
    @Test
    fun `evaluate basic function`() {

        val tf = Graph()

        val inputs = tf.constant(2.0)

        val weights = tf.get_variable("weights", tf.constant(3.0))
        val biases = tf.get_variable("bias", tf.constant(5.0))

        val graph1 = tf.mult(weights, inputs) // 3*2
        assertEquals(Tensor(3.0 * 2.0), graph1())

        val graph2 = tf.mult(weights) // *3(2)
        assertEquals(Tensor(2.0 * 3.0), graph2(inputs))

        val graph3 = tf.add(biases) // +5(2)
        assertEquals(Tensor(2.0 + 5.0), graph3(inputs))

        val graph35 = tf.relu(tf.add(biases)) // ÷3(+5(2))
        assertEquals(Tensor((2.0 + 5.0) / 3), graph35(inputs))

        val graph6 = tf.add(biases, biases) // 5+5
        assertEquals(Tensor(5.0 + 5.0), graph6())

        val graph4 = tf.add(biases, tf.mult(weights)) // 5 + (*3)(2)
        assertEquals(Tensor(5.0 + 2.0 * 3.0), graph4(inputs))

        val graph5 = tf.add(tf.mult(weights), biases) // (*3)(2) + 5
        assertEquals(Tensor(2.0 * 3.0 + 5.0), graph5(inputs))

        val graph7 = tf.relu(tf.add(tf.mult(weights), biases)) // ÷3(*3(2) + 5.0)
        assertEquals(Tensor((2.0 * 3.0 + 5.0) / 3.0), graph7(inputs))


        // Second layer
        val weights2 = tf.get_variable("weights2", tf.constant(3.0))
        val biases2 = tf.get_variable("bias2", tf.constant(5.0))

        val graph20 = tf.relu(tf.add(tf.mult(graph7, weights2), biases2)) // ÷3(g7(2.0) * 3.0 + 5.0)
        assertEquals(Tensor((graph7(inputs).value * 3.0 + 5.0) / 3), graph20(inputs))

        val inputs2 = tf.constant(1.0) // ÷3(g7(1.0) * 3.0 + 5.0)
        assertEquals(Tensor(13.0 / 3), graph20(inputs2))

        // Third layer
        val graph30 = tf.relu(tf.add(tf.mult(weights), biases2)) // ÷3(16+5)
        assertEquals(Tensor((16.0 + 5.0) / 3), graph30({ graph20(inputs) }))

        // Composite layers
        val fc = tf.fc(weights, biases, weights2, biases2) // ÷3(÷3(3(2)+5)*3+5)
        assertEquals(Tensor((3.0 * (3.0 * 2.0 + 5.0) / 3 + 5.0) / 3), fc(inputs))

        val graph50 = tf.relua(inputs)
        println("Graph: " + graph50.javaClass.declaredMethods[1])
        println("Relu: " + (Graph::relua).javaMethod)


        println(tf.del(graph50))
        println(tf.del(graph50)?.invoke(graph20(inputs2)))
    }
}

class Graph {
    /** Nullary operators */
    fun constant(x: Double): NullaryOp {
        return { Tensor(x) }
    }

    fun placeholder(): NullaryOp {
        return { Tensor(0.0) }
    }

    fun get_variable(name: String, initialValue: NullaryOp): () -> Variable {
        return { Variable(initialValue().value) }
    }

    /** Binary operators */

    fun mult(lhs: NullaryOp) = { rhs: NullaryOp -> mult(lhs(), rhs()) }

    fun mult(lhs: NullaryOp, rhs: NullaryOp) = { mult(lhs(), rhs()) }

    fun mult(lhs: NullaryOp, rhs: (NullaryOp) -> Tensor) = { x: NullaryOp -> mult(lhs(), rhs(x)) }
    fun mult(lhs: (NullaryOp) -> Tensor, rhs: NullaryOp) = { x: NullaryOp -> mult(lhs(x), rhs()) }

    fun mult(lhs: Tensor, rhs: Tensor): Tensor = Tensor(lhs.value * rhs.value)


    /** curried binary operator */
    fun add(lhs: NullaryOp) = { rhs: NullaryOp -> add(lhs(), rhs()) }

    /** Intermediate operators */
    fun add(lhs: NullaryOp, rhs: NullaryOp) = { add(lhs(), rhs()) }

    fun add(lhs: NullaryOp, rhs: (NullaryOp) -> Tensor) = { x: NullaryOp -> add(lhs(), rhs(x)) }
    fun add(lhs: (NullaryOp) -> Tensor, rhs: NullaryOp) = { x: NullaryOp -> add(lhs(x), rhs()) }

    /** Actual evaluation */
    fun add(lhs: Tensor, rhs: Tensor): Tensor = Tensor(lhs.value + rhs.value)


    /** Intermediate operators */
    fun relua(f: NullaryOp) = { relue(f()) }

    fun relu(f: (NullaryOp) -> Tensor) = { x: NullaryOp -> relue(f(x)) }

    fun reluprimea(f: NullaryOp) = { reluprimee(f()) }

    /** Actual evaluation */
    fun relue(x: Tensor) = if (x.value < 0) Tensor(0.0) else Tensor(x.value / 3)

    fun reluprimee(x: Tensor) = if (x.value < 0) Tensor(0.00001) else Tensor(1.000001)

    /** composite functions */
    fun slp(weights: NullaryOp, biases: NullaryOp) = relu(add(mult(weights), biases))

    fun fc(w1: NullaryOp, b1: NullaryOp, w2: NullaryOp, b2: NullaryOp) = relu(add(mult(w2, slp(w1, b1)), b2))

    private val map = mapOf(Pair(Graph::relua.javaMethod, Graph::reluprimea.javaMethod))

    fun del(f: NullaryOp): UnaryOp? {
        val regex = Regex("public final (.*) (.*)\\$(.*)\\$1\\.invoke\\(\\)")
        val (w, rt, ct, m) = regex.find(f.javaClass.declaredMethods[1].toString())!!.groupValues
        val dm = Class.forName(ct).getDeclaredMethod(m, kotlin.jvm.functions.Function0::class.javaObjectType)
        return { dx: Tensor ->
            (method(dm)?.invoke(this@Graph, { dx }) as NullaryOp)()
        }
    }

    private fun method(dm: Method?) = map.get(dm)
}

/** Types */
data class Tensor(val value: Double)
typealias Variable = Tensor
typealias NullaryOp = () -> Tensor
typealias UnaryOp = (Tensor) -> Tensor
typealias Functor = (NullaryOp) -> NullaryOp

package experiment3

import org.junit.Test
import kotlin.test.assertEquals

class FunctionalGraph3Should {

    @Test
    fun `evaluate basic function`() {
        val func = square() // x^2
        assertEquals(2.0 * 2.0, func(2.0))
    }

    @Test
    fun `compose basic functions`() {
        val func = square{square()} // (x^2)^2 = x^4
        assertEquals(2.0 * 2.0 * 2.0 * 2.0, func(2.0))
    }


    @Test
    fun `compose simple functions`() {
        val composite = square('x')(square('x'))
        assertEquals(2.0 * 2.0 * 2.0 * 2.0, composite.first(2.0))
        assertEquals(3.0 * 3.0 * 3.0 * 3.0, composite.first(3.0))
    }

    @Test
    fun `differentiate simple functions`() {
        val func = square('x') // x^2
        assertEquals(0.0, func.div('y')(2.0))
        assertEquals(2.0 * 3.0, func.div('x')(3.0))
    }

    @Test
    fun `differentiate composite functions`() {
        val composite = square('x')(square('x')) // f: x -> (x^2)^2 = x^4    df/dx=4*x^3
        assertEquals(0.0, composite.div('y')(3.0))
        assertEquals(4.0 * 3.0 * 3.0 * 3.0, composite.div('x')(3.0))
    }

    @Test
    fun `differentiate composite functions2`() {
        val composite = pow(3, 'x')(square('x')) // f: x -> (x^2)^3 = x^6    df/dx=6*x^5
        assertEquals(0.0, composite.div('y')(3.0))
        assertEquals(6.0 * Math.pow(2.5, 5.0), composite.div('x')(2.5))
        assertEquals(6.0 * 3.0 * 3.0 * 3.0 * 3.0 * 3.0, composite.div('x')(3.0))
    }
}

fun times(f: UnaryOp, g: UnaryOp): BinaryOp = { x, y -> times(f(x), g(y)) }
fun times(x: Double, y: Double): Double = x * y


fun square(sym: Char = 'x'): FuncDef = Triple({ x -> x * x }, sym, { x -> 2.0 * x })
fun square(function: FuncDef) : FuncDef = square().invoke(function)
fun square(function: () -> FuncDef) : FuncDef = square(function())

fun pow(n: Int, sym: Char = 'x'): FuncDef = Triple({ x -> pow(n, x) }, sym, { x -> n * pow(n - 1, x) })
fun pow(n: Int, x: Double): Double = Math.pow(x, n.toDouble())


operator fun FuncDef.invoke(f: FuncDef): FuncDef = Triple(
        { x -> this.first(f.first(x)) }, // composition
        this.second, // symbol, TODO("should aggregate symbols")
        { x -> times(this.div()(f.first(x)), f.div()(x)) }  // chain rule
)

operator fun FuncDef.invoke(x: Double): Double = this.first(x)

fun FuncDef.div(symX: Char = this.second): UnaryOp = when (symX) {
    this.second -> this.third
    else -> { x -> 0.0 }
}


typealias UnaryOp = (Double) -> Double
typealias BinaryOp = (Double, Double) -> Double

typealias FuncDef = Triple<UnaryOp, Char, UnaryOp>
typealias BinaryFuncDef = Triple<BinaryOp, Pair<Char, Char>, (Double) -> Pair<Double, Double>>
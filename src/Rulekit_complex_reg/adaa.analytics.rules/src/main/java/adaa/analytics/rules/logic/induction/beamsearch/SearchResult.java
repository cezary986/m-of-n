package adaa.analytics.rules.logic.induction.beamsearch;

import org.jetbrains.annotations.NotNull;

public class SearchResult<T> implements Comparable<SearchResult> {
    public T value;
    public Comparable evaluation;
    public SearchResult<T> parentSolution;

    public SearchResult(T value, Comparable evaluation) {
        this.value = value;
        this.evaluation = evaluation;
        this.parentSolution = null;
    }

    public SearchResult(T value, Double evaluation, SearchResult<T> parentSolution) {
        this.value = value;
        this.evaluation = evaluation;
        this.parentSolution = parentSolution;
    }

    @Override
    public int compareTo(@NotNull SearchResult o) {
        if (evaluation == null && o.evaluation != null)
            return -1;
        else if (evaluation == null && o.evaluation == null)
            return 0;
        else if (evaluation != null && o.evaluation == null)
            return 1;
        else
            return evaluation.compareTo(o.evaluation);
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "value=" + value +
                ", evaluation=" + evaluation +
                '}';
    }
}

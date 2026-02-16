import { useState, useEffect } from "react";
import Filter from "./Filters";
import ProductCard from "./ProductCard";
import qs from "query-string";
import { Filter as FilterIcon, ChevronLeft } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import { useLanguage } from "../../contexts/LanguageContext";
import { client } from "../../services";

interface Rug {
  price: number | undefined;
  id: number;
  catalog: { id: number; name: string }; // Updated to include catalog object
  name: string;
  image: string;
  collection_type: number | string;
  style?: number;
  room?: number;
  color?: number;
  shape?: number;
}

interface FilterOption {
  id: number;
  name: string;
}

interface FilterLabels {
  catalog: string;
  collections: string;
  styles: string;
  rooms: string;
  colors: string;
  shapes: string;
}

interface FilterOptions {
  catalog: { id: number; name: string };
  collections: FilterOption[];
  styles: FilterOption[];
  rooms: FilterOption[];
  colors: FilterOption[];
  shapes: FilterOption[];
  labels: FilterLabels;
}

const Catalog = () => {
  const { t, language } = useLanguage();
  const location = useLocation();
  const collectionId = location.state?.collectionId;
  const { id, categoryId } = useParams<{ id: string; categoryId: string }>();
  const [rugData, setRugData] = useState<Rug[]>([]);
  const name = location.state?.name;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(
    () => typeof window !== "undefined" && window.innerWidth >= 1024,
  );
  const [currentPage, setCurrentPage] = useState(1);
  const [sortOption, setSortOption] = useState("4");
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    catalog: { id: 0, name: "" },
    collections: [],
    styles: [],
    rooms: [],
    colors: [],
    shapes: [],
    labels: {
      catalog: "",
      collections: "",
      styles: "",
      rooms: "",
      colors: "",
      shapes: "",
    },
  });
  const [filters, setFilters] = useState({
    collection: [] as number[],
    style: [] as number[],
    room: [] as number[],
    color: [] as number[],
    shape: [] as number[],
  });

  const itemsPerPage = 21;

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const handleResize = () => {
      setShowFilters(window.innerWidth >= 1024);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    const lang = mapLang(language);

    const fetchFilterOptions = async () => {
      try {
        const res = await client.get(
          `/${lang}/api/v1/catalog/filter_choices_for_carpet_model/${categoryId}/`,
        );
        setFilterOptions({
          catalog: res.data.catalog || { id: 0, name: "" },
          collections: res.data.collections || [],
          styles: Array.isArray(res.data.styles)
            ? Array.isArray(res.data.styles[0])
              ? res.data.styles[0]
              : res.data.styles
            : [],
          rooms: res.data.rooms || [],
          colors: res.data.colors || [],
          shapes: res.data.shapes || [],
          labels: res.data.labels || {
            catalog: "",
            collections: "",
            styles: "",
            rooms: "",
            colors: "",
            shapes: "",
          },
        });
      } catch (err) {
        console.error(`Error fetching filter options for catalog ${id}:`, err);
        setError(t("") || "");
      }
    };

    const fetchAllRugs = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await client.get(
          `/${lang}/api/v1/catalog/get_carpet_models_by_carpet_id/${id}/`,
        );
        setRugData(res.data);
      } catch (err) {
        console.error("Error fetching all rugs:", err);
        setError(t("catalog.error.rugs") || "Mahsulotlarni yuklashda xatolik");
      } finally {
        setLoading(false);
      }
    };

    const fetchFilteredRugs = async () => {
      try {
        setLoading(true);
        setError(null);
        const queryParams = qs.stringify(
          {
            sort_by: sortOption !== "4" ? sortOption : undefined,
            collections:
              filters.collection.length > 0
                ? filters.collection.join(",")
                : undefined,
            styles:
              filters.style.length > 0 ? filters.style.join(",") : undefined,
            rooms: filters.room.length > 0 ? filters.room.join(",") : undefined,
            colors:
              filters.color.length > 0 ? filters.color.join(",") : undefined,
            shapes:
              filters.shape.length > 0 ? filters.shape.join(",") : undefined,
          },
          { skipNull: true, skipEmptyString: true },
        );

        const res = await client.get(
          `/${lang}/api/v1/catalog/filter_and_sort_carpets_for_carpet_model/${categoryId}/?${queryParams}`,
        );
        setRugData(res.data);
      } catch (err) {
        console.error("Error fetching filtered rugs:", err);
        setError(
          t("catalog.error.rugs") ||
            "Filtrlangan mahsulotlarni yuklashda xatolik",
        );
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchFilterOptions();
      const noFiltersApplied =
        filters.collection.length === 0 &&
        filters.style.length === 0 &&
        filters.room.length === 0 &&
        filters.color.length === 0 &&
        filters.shape.length === 0 &&
        sortOption === "4";

      if (noFiltersApplied) {
        fetchAllRugs();
      } else {
        fetchFilteredRugs();
      }
    }
  }, [id, language, filters, sortOption, categoryId]);

  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
    setCurrentPage(1);
  };

  const clearFilters = () => {
    setFilters({
      collection: [],
      style: [],
      room: [],
      color: [],
      shape: [],
    });
  };

  const mapCollectionType = (type: number | string): string => {
    if (typeof type === "string") {
      return type;
    }
    const collectionMap: { [key: number]: string } = {
      1: "New",
      2: "Sale",
      3: "Hit",
      4: "Default",
    };
    return collectionMap[type] || "";
  };

  const totalPages = Math.ceil(rugData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentRugs = rugData.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="bg-[#FFFCE0] md:pt-28 pt-24">
      <div className="container mx-auto px-4 py-6">
        {error && (
          <div className="mb-4 p-4 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}
        <div className="mb-6">
          <div className="flex md:mb-12 mb-7 items-center text-base text-gray-600">
            <ChevronLeft size={20} className="text-gray600" />
            <Link to="/">{t("nav.home")}</Link>
            <div
              className="pl-3 flex items-center cursor-pointer"
              onClick={() => window.history.back()}
            >
              <ChevronLeft size={20} className="text-gray-600" />
              {rugData[0]?.catalog?.name}
            </div>
            <div className="pl-3 flex items-center font-semibold cursor-pointer">
              <ChevronLeft size={20} className="text-gray-600" />
              {name}
            </div>
          </div>

          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg w-fit"
            >
              <FilterIcon size={16} />
              <span>{t("catalog.filter")}</span>
            </button>

            <div className="flex overflow-x-auto gap-4 pb-2">
              {[
                { label: t("catalog.all"), value: "4" },
                { label: t("catalog.new"), value: "1" },
                { label: t("catalog.bestseller"), value: "2" },
                { label: t("catalog.sale"), value: "3" },
              ].map(({ label, value }) => (
                <button
                  key={value}
                  onClick={() => setSortOption(value)}
                  className={`px-4 py-2 whitespace-nowrap border-b-2 transition-colors ${
                    sortOption === value
                      ? "text-gray-800 font-semibold border-gray-800"
                      : "text-gray-600 border-transparent hover:border-gray-300"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-6">
          {showFilters && (
            <div className="md:w-80">
              <Filter
                filters={filters}
                onFilterChange={handleFilterChange}
                onClearFilters={clearFilters}
                filterOptions={filterOptions}
                initialCollectionId={collectionId}
              />
            </div>
          )}

          <div
            className={`${
              showFilters ? "md:w-[calc(100%-320px)]" : "w-full min-h-[500px]"
            }`}
          >
            {loading ? (
              <p className="text-center py-12">{t("common.loading") || ""}</p>
            ) : rugData.length > 0 ? (
              <>
                <div
                  className={`grid grid-cols-2 sm:grid-cols-2 ${
                    showFilters ? "lg:grid-cols-3" : "lg:grid-cols-4"
                  } gap-8`}
                >
                  {currentRugs.map((rug) => {
                    return (
                      <ProductCard
                        key={rug.id}
                        id={rug.id}
                        name={rug.name}
                        image={rug.image}
                        price={rug.price}
                        collection_type={mapCollectionType(rug.collection_type)}
                      />
                    );
                  })}
                </div>

                {totalPages > 1 && (
                  <div className="flex justify-center items-center gap-2 mt-8">
                    <button
                      onClick={() =>
                        setCurrentPage(Math.max(1, currentPage - 1))
                      }
                      disabled={currentPage === 1}
                      className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                    >
                      ←
                    </button>

                    {(() => {
                      const visiblePages = 5;
                      const halfVisible = Math.floor(visiblePages / 2);
                      let startPage = Math.max(1, currentPage - halfVisible);
                      let endPage = Math.min(
                        totalPages,
                        currentPage + halfVisible,
                      );

                      if (endPage - startPage < visiblePages - 1) {
                        startPage = Math.max(1, endPage - (visiblePages - 1));
                      }

                      const pages = [];
                      if (startPage > 1) {
                        pages.push(
                          <button
                            key={1}
                            onClick={() => setCurrentPage(1)}
                            className={`px-3 py-2 rounded ${
                              currentPage === 1
                                ? "bg-[#D7CCC8] text-white"
                                : "text-gray-600 hover:bg-gray-100"
                            }`}
                          >
                            1
                          </button>,
                        );
                        if (startPage > 2)
                          pages.push(
                            <span key="start-ellipsis" className="px-2">
                              ...
                            </span>,
                          );
                      }

                      for (let i = startPage; i <= endPage; i++) {
                        pages.push(
                          <button
                            key={i}
                            onClick={() => setCurrentPage(i)}
                            className={`px-3 py-2 rounded ${
                              currentPage === i
                                ? "bg-[#D7CCC8] text-white"
                                : "text-gray-600 hover:bg-gray-100"
                            }`}
                          >
                            {i}
                          </button>,
                        );
                      }

                      if (endPage < totalPages) {
                        if (endPage < totalPages - 1)
                          pages.push(
                            <span key="end-ellipsis" className="px-2">
                              ...
                            </span>,
                          );
                        pages.push(
                          <button
                            key={totalPages}
                            onClick={() => setCurrentPage(totalPages)}
                            className={`px-3 py-2 rounded ${
                              currentPage === totalPages
                                ? "bg-[#D7CCC8] text-white"
                                : "text-gray-600 hover:bg-gray-100"
                            }`}
                          >
                            {totalPages}
                          </button>,
                        );
                      }

                      return pages;
                    })()}

                    <button
                      onClick={() =>
                        setCurrentPage(Math.min(totalPages, currentPage + 1))
                      }
                      disabled={currentPage === totalPages}
                      className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                    >
                      →
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12">
                <p className="text-gray-500">
                  {t("catalog.noProducts") || "Mahsulot topilmadi"}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Catalog;

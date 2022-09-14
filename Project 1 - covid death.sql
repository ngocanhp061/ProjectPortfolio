-- Feel free to visit my Analysis via: https://panoramic-zebra-edbbe.stackbit.app/blog/post-four/
-- Feel free to visit my tableau dashboard via: https://public.tableau.com/app/profile/pham.ngoc.anh1751/viz/CovidDashboardPortfolioProject_16625348695530/Dashboard1?publish=yes

Create database ProjectPortfolio

Use [ProjectPortfolio]
Go

Select * into CovidDeaths1 from CovidDeaths where continent is not null order by 3,4
Select * into CovidVaccinations1 from CovidVaccinations where continent is not null order by 3,4

--Select Data that we are going to be using:
Select location, date, total_cases, new_cases, total_deaths, population
from CovidDeaths1
order by 1,2

--Looking at Total cases vs Total Deaths
--shows likelihood of dying if you contract covid in your country
Select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
from CovidDeaths1
Where location like '%Vietnam%'
order by 1,2
--(Tính đến 30/4/2021, tỉ lệ mắc covid là 1.195% dân số) 

--Looking at Total cases vs Populations
--shows what percentage of population got covid 
Select location, date, total_cases, population, (total_cases/population)*100 as CasesPercentage
from CovidDeaths1
Where location like '%Vietnam%'
order by 1,2
--(Tính đến 30/4/2021, tỉ lệ dân số mắc covid là 0.003%)

--Looking at Countries with Highest Infection Rate compared to Population
Select location, population, max(total_cases) as HighestInfectionCount, max((total_cases/population))*100 as CasesPercentage
from CovidDeaths1
--Where location like '%Vietnam%'
group by location, population
order by CasesPercentage desc

--Showing Countries with Highest Death Count per Population
/*Phát hiện ra datatype bị sai nên nó đếm sai kqua -> Dùng cast để đổi nó về dạng datatype đúng là int*/
/*Phát hiện ra cột continent bị đổi sang cột location, dẫn đến cột continent bị null -> quay về bên trên và fix*/
Select location, max(cast(Total_deaths as int)) as TotalDeathCount
from CovidDeaths1
--Where location like '%Vietnam%'
group by location
order by TotalDeathCount desc

--Let's break things down by continent
Select location, max(cast(Total_deaths as int)) as TotalDeathCount
from CovidDeaths
--Where location like '%Vietnam%'
where continent is null 
group by location
order by TotalDeathCount desc

/*Số liệu not perfect vì số liệu vẫn còn ở những ô continent bị null*/

Select continent, max(cast(Total_deaths as int)) as TotalDeathCount
from CovidDeaths1
--Where location like '%Vietnam%'
group by continent
order by TotalDeathCount desc

--Showing continent with the highest death count per population
Select continent, max(cast(Total_deaths as int)) as TotalDeathCount
from CovidDeaths1
--Where location like '%Vietnam%'
group by continent
order by TotalDeathCount desc

--Global numbers
/*muốn chỉ group by date, chứ ko group by total_cases và total_deaths,
vì thế dùng hàm sum(new_cases) để ra total_cases hàng ngày */
Select date, sum(new_cases) as total_cases, sum(cast(new_deaths as int)) as total_deaths,
sum(cast(new_deaths as int))/sum(new_cases)*100 as DeathPercentage
from CovidDeaths1
--Where location like '%Vietnam%'
group by date
order by 1,2 /*(Này là dữ liệu theo ngày)*/

Select sum(new_cases) as total_cases, sum(cast(new_deaths as int)) as total_deaths,
sum(cast(new_deaths as int))/sum(new_cases)*100 as DeathPercentage
from CovidDeaths1
--Where location like '%Vietnam%'
order by 1,2 /*(Này là dữ liệu tổng tính đến ngày 30/4/2021)*/


--Looking at Total Population vs Vaccinations
--Create CTE
With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, rolling_total_vaccinations)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(cast(vac.new_vaccinations as int)) over (partition by dea.location order by dea.location, dea.date) as rolling_total_vaccinations
from CovidDeaths dea
join CovidVaccinations vac
on dea.location = vac.location
and dea.date = vac.date
where dea.continent is not null 
--order by 2,3
)
Select *, (rolling_total_vaccinations/population)*100
from PopvsVac
order by 2,3

--Create new table for later use
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(cast(vac.new_vaccinations as int)) over (partition by dea.location order by dea.location, dea.date) as rolling_total_vaccinations
into PercentPopulationVaccinated
from CovidDeaths dea
join CovidVaccinations vac
on dea.location = vac.location
and dea.date = vac.date
--where dea.continent is not null 
--order by 2,3

Select *, (rolling_total_vaccinations/population)*100
from PercentPopulationVaccinated
order by 2,3

--Create View to store data for later use 
Create View VPercentPopulationVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(cast(vac.new_vaccinations as int)) over (partition by dea.location order by dea.location, dea.date) as rolling_total_vaccinations
from CovidDeaths dea
join CovidVaccinations vac
on dea.location = vac.location
and dea.date = vac.date
where dea.continent is not null 
--order by 2,3

Select * from VPercentPopulationVaccinated
